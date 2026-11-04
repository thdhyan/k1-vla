import os

from k1_chat import debug


def test_meminfo_parses_mb():
    m = debug.meminfo("MemTotal: 8000000 kB\nMemAvailable: 204800 kB\nSwapTotal: 2048 kB\nSwapFree: 1024 kB\n")
    assert m == {"total_mb": 7812, "available_mb": 200, "swap_used_mb": 1, "swap_total_mb": 2}


def test_note_shows_up_in_collect(monkeypatch):
    for k in ("K1_CHAT_MODEL", "K1_VISION_MODEL"):
        monkeypatch.delenv(k, raising=False)
    debug.note("tts", 1.234, detail="12 chars")
    d = debug.collect()
    assert d["timings"]["tts"]["secs"] == 1.23 and d["timings"]["tts"]["ok"]
    assert d["chat_llm"] == {"configured": False}
    assert d["memory"]["total_mb"] > 0
    assert any(p["pid"] == os.getpid() for p in debug.processes(top=10_000)["top"])
    assert {w["name"] for w in d["processes"]["watched"]} >= {"nemo-asrd", "walk deploy"}


def test_failing_probe_reports_error_not_raise():
    def boom():
        raise RuntimeError("x")
    assert debug._cached("t_boom", 0, boom) == {"error": "RuntimeError: x"}
