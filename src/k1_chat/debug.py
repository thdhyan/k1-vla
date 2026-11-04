"""Health snapshot for the console's System panel (GET /api/debug). tests/test_debug.py.

Memory/swap, GPU load, OOM kills this boot, nemo-asrd (systemd + /health), the robot's own audio service
(must be OFFLINE while we hold the mic), camera node, OpenRouter key/limits, last ASR/TTS/LLM latency
(note() from robot_audio._timed and llm.complete), and the processes that matter for k1_chat on 8 GB.
Every probe is best effort: a failing probe reports its error, never raises. Slow probes are cached."""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

PROC = Path("/proc")
AUDIO_API = "http://127.0.0.1:20001/service/status"
GPU_LOAD = Path("/sys/devices/platform/gpu.0/load")  # Orin: 0..1000
# name -> cmdline regex; anything memory-heavy that competes with nemo/kokoro on the 8 GB Orin
WATCH = {
    "k1_chat agent": r"k1_chat\.robot_audio",
    "nemo-asrd": r"nemo-asrd\.py",
    "walk deploy": r"k1_gamepad_deploy|k1_mjlab_deploy",
    "booster motion": r"/container/motion$",
    "camera node": r"cam_stream_receiver",
    "video stream": r"booster-video-stream",
    "pulseaudio": r"pulseaudio",
    "desktop (gdm)": r"gnome-shell|Xorg|gdm",
}

_lock = threading.Lock()
_timings: dict[str, dict] = {}
_cache: dict[str, tuple[float, object]] = {}


def note(name: str, secs: float, ok: bool = True, detail: str = "") -> None:
    """Record the latest latency of a pipeline stage (asr, tts, llm, ...)."""
    with _lock:
        _timings[name] = {"secs": round(secs, 2), "ok": ok, "detail": detail[:160], "at": time.time()}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception as e:  # noqa: BLE001 — a probe must never break the console
        val = {"error": f"{type(e).__name__}: {e}"}
    _cache[key] = (now, val)
    return val


def _http_json(url: str, timeout: float = 1.5, data: bytes | None = None, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def meminfo(text: str | None = None) -> dict:
    text = text if text is not None else (PROC / "meminfo").read_text()
    kb = {k: int(v) for k, v in re.findall(r"^(\w+):\s+(\d+)", text, re.M)}
    mb = lambda k: kb.get(k, 0) // 1024  # noqa: E731
    return {"total_mb": mb("MemTotal"), "available_mb": mb("MemAvailable"),
            "swap_used_mb": mb("SwapTotal") - mb("SwapFree"), "swap_total_mb": mb("SwapTotal")}


def processes(top: int = 6) -> dict:
    hz, boot = os.sysconf("SC_CLK_TCK"), time.time() - float((PROC / "uptime").read_text().split()[0])
    rows = []
    for p in PROC.iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            rss = int(re.search(r"VmRSS:\s+(\d+)", (p / "status").read_text()).group(1)) // 1024
            start = int((p / "stat").read_text().rsplit(")", 1)[1].split()[19]) / hz
        except (OSError, AttributeError, IndexError, ValueError):
            continue  # kernel thread or exited
        if cmd:
            rows.append({"pid": int(p.name), "rss_mb": rss, "age_s": int(time.time() - boot - start), "cmd": cmd[:140]})
    watched = []
    for name, rx in WATCH.items():
        hits = [r for r in rows if re.search(rx, r["cmd"])]
        watched.append({"name": name, "count": len(hits), "rss_mb": sum(r["rss_mb"] for r in hits),
                        "pids": [r["pid"] for r in hits][:6]})
    return {"watched": watched, "top": sorted(rows, key=lambda r: -r["rss_mb"])[:top]}


def _systemd(unit: str) -> dict:
    out = subprocess.run(["systemctl", "show", unit, "-p", "ActiveState", "-p", "NRestarts"],
                         capture_output=True, text=True, timeout=2).stdout
    return dict(line.split("=", 1) for line in out.split())


def _nemo() -> dict:
    url = os.environ.get("K1_NEMOTRON_URL", "http://127.0.0.1:8090/transcribe").rsplit("/", 1)[0]
    out = {"systemd": _cached("nemo_sd", 5, lambda: _systemd("nemo-asrd")), "url": url}
    try:
        h = _http_json(url + "/health")
        out.update(ok=bool(h.get("ready")), load_s=h.get("load_s"), model=h.get("model"))
    except Exception as e:  # noqa: BLE001
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


def _robot_audio() -> dict:
    r = _http_json(AUDIO_API, data=b'{"service":"audio"}', headers={"content-type": "application/json"})
    stage = r.get("services", {}).get("audio", {}).get("stage")
    return {"stage": stage, "ok": stage == "OFFLINE", "note": "OFFLINE = mic free for k1_chat"}


def _ooms() -> dict:
    out = subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-o", "short-iso", "-g", "Killed process"],
                         capture_output=True, text=True, timeout=3).stdout
    lines = [ln for ln in out.splitlines() if "Killed process" in ln]
    return {"count": len(lines), "last": [re.sub(r"total-vm.*", "", ln)[:120] for ln in lines[-3:]]}


def _llm_endpoint(prefix: str) -> dict:
    model = os.environ.get(f"{prefix}_MODEL")
    if not model:
        return {"configured": False}
    base = os.environ.get(f"{prefix}_BASE_URL")
    if base:  # self-hosted (laptop llm_panel / vLLM)
        try:
            ids = [m.get("id") for m in _http_json(base.rstrip("/") + "/models", timeout=3).get("data", [])]
            return {"model": model, "base": base, "ok": True, "served": ids}
        except Exception as e:  # noqa: BLE001
            return {"model": model, "base": base, "ok": False, "error": f"{type(e).__name__}: {e}"}
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {"model": model, "ok": False, "error": "OPENROUTER_API_KEY not set"}
    t = time.time()
    d = _http_json("https://openrouter.ai/api/v1/key", timeout=5, headers={"Authorization": f"Bearer {key}"}).get("data", {})
    return {"model": model, "base": "openrouter", "ok": True, "rtt_s": round(time.time() - t, 2),
            "free_tier": d.get("is_free_tier"), "usage": d.get("usage"), "limit": d.get("limit"),
            "rate_limit": d.get("rate_limit")}


def collect() -> dict:
    gpu = _cached("gpu", 1, lambda: int(GPU_LOAD.read_text()) / 10) if GPU_LOAD.exists() else None
    with _lock:
        timings = {k: {**v, "ago_s": int(time.time() - v["at"])} for k, v in _timings.items()}
    return {
        "t": time.time(),
        "memory": _cached("mem", 1, meminfo),
        "gpu_load_pct": gpu,
        "oom_kills_this_boot": _cached("oom", 15, _ooms),
        "nemo_asr": _cached("nemo", 2, _nemo),
        "robot_audio_service": _cached("audio", 5, _robot_audio),
        "chat_llm": _cached("llm_chat", 30, lambda: _llm_endpoint("K1_CHAT")),
        "vision_llm": _cached("llm_vision", 30, lambda: _llm_endpoint("K1_VISION")),
        "asr_backend": os.environ.get("K1_CHAT_ASR", "nemo"),
        "timings": timings,
        "processes": _cached("procs", 2, processes),
    }
