"""Laptop GPU ASR with the nemo-asrd HTTP API, so robots offload speech recognition and keep their RAM.

  laptop:  PYTHONPATH=src uv run --with faster-whisper --with nvidia-cublas-cu12 --with "nvidia-cudnn-cu12==9.*" \
             python -m k1_chat.asr_server --model large-v3-turbo        (listens on 0.0.0.0:8090)
  robot:   K1_CHAT_ASR=nemo K1_NEMOTRON_URL=http://192.168.50.250:8090/transcribe  (k1voice.sh passes it)

  GET  /health      {"ready": 1, "model": ..., "load_s": ...}
  POST /transcribe  {"pcm16": base64 int16 mono, "sr": 16000, "lang": "en"} -> {"text": ..., "secs": ...}
tests/test_asr_server.py."""
import argparse
import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


def make_server(host: str, port: int, transcribe: Callable[[bytes, str], str], info: dict) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json({"ready": 1, **info}) if self.path == "/health" else self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/transcribe":
                return self._json({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if int(req.get("sr", 16000)) != 16000:
                    return self._json({"error": "sr must be 16000"}, 400)
                pcm = base64.b64decode(req["pcm16"])
            except (ValueError, KeyError) as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            t = time.time()
            text = transcribe(pcm, req.get("lang", "en"))
            secs = time.time() - t
            print(f"[asr] {len(pcm) / 32000:.1f}s audio -> {secs:.2f}s {text!r}", flush=True)
            self._json({"text": text, "secs": round(secs, 3)})

    return ThreadingHTTPServer((host, port), H)


def whisper(model: str, device: str):
    import threading

    import numpy as np
    from faster_whisper import WhisperModel

    m = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")
    lock = threading.Lock()  # one GPU decode at a time

    def transcribe(pcm: bytes, lang: str) -> str:
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        with lock:
            segs, _ = m.transcribe(audio, language=lang, beam_size=1, vad_filter=False,
                                   condition_on_previous_text=False)
            return " ".join(s.text for s in segs).strip()

    transcribe(b"\0\0" * 16000, "en")  # warm up CUDA kernels
    return transcribe


def _cuda_libs() -> None:
    """pip nvidia-cublas/cudnn wheels are not on the loader path; ctranslate2 dlopens them, so preload."""
    import ctypes
    import glob
    import site
    for pat in ("nvidia/cublas/lib/libcublas*.so.*", "nvidia/cudnn/lib/libcudnn*.so.*"):
        for d in site.getsitepackages():
            for f in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args(argv)
    if a.device == "cuda":
        _cuda_libs()
    t = time.time()
    fn = whisper(a.model, a.device)
    info = {"model": f"faster-whisper {a.model} ({a.device})", "load_s": round(time.time() - t, 1)}
    print(f"asr_server ready on {a.host}:{a.port} {info}", file=sys.stderr, flush=True)
    make_server(a.host, a.port, fn, info).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
