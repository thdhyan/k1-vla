"""NemotronAsr adapter -> nemo-asrd daemon (K1_CHAT_ASR=nemo). Tests first: plan
WAKEWORD_NEMO.md §3.3 / booster_ws PLAN_VOICE_VLM §9.6."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from k1_chat.robot_audio import NemotronAsr


class _FakeDaemon(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        _FakeDaemon.seen.append(req)
        body = json.dumps({"text": "hello nemotron", "wall_s": 0.12}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = json.dumps({"ready": 1}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def daemon():
    _FakeDaemon.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeDaemon)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/transcribe", _FakeDaemon
    srv.shutdown()


def _pcm(seconds: float = 0.5) -> bytes:
    return b"\x00\x01" * int(16000 * seconds)


def test_roundtrip_sends_pcm_lang_sr(daemon):
    url, fake = daemon
    asr = NemotronAsr(url=url, lang="en")
    assert asr(_pcm()) == "hello nemotron"
    req = fake.seen[0]
    assert req["lang"] == "en"
    assert req["sr"] == 16000
    import base64

    assert base64.b64decode(req["pcm16"]) == _pcm()


def test_daemon_down_returns_empty():
    # port 9 (discard) refused on loopback -> adapter must swallow and return ""
    asr = NemotronAsr(url="http://127.0.0.1:9/transcribe", lang="en", timeout=2.0)
    assert asr(_pcm()) == ""


def test_short_pcm_skipped_without_http(daemon):
    url, fake = daemon
    asr = NemotronAsr(url=url)
    assert asr(b"\x00\x01") == ""  # 1 sample < 100 ms
    assert fake.seen == []


def test_factory_selects_backend(monkeypatch):
    from k1_chat import robot_audio as ra

    monkeypatch.setattr(ra, "WhisperAsr", lambda: "WHISPER_SENTINEL")
    monkeypatch.setenv("K1_CHAT_ASR", "whisper")
    assert ra.make_asr() == "WHISPER_SENTINEL"

    url, _ = None, None  # noqa: F841 — clarity
    with daemon_up() as durl:
        monkeypatch.setenv("K1_CHAT_ASR", "nemo")
        monkeypatch.setenv("K1_NEMOTRON_URL", durl)
        asr = ra.make_asr()
        assert isinstance(asr, ra.NemotronAsr)


def test_factory_falls_back_when_daemon_dead(monkeypatch):
    from k1_chat import robot_audio as ra

    monkeypatch.setattr(ra, "WhisperAsr", lambda: "WHISPER_SENTINEL")
    monkeypatch.setenv("K1_CHAT_ASR", "nemo")
    monkeypatch.setenv("K1_NEMOTRON_URL", "http://127.0.0.1:9/transcribe")
    assert ra.make_asr() == "WHISPER_SENTINEL"


class daemon_up:
    """Context manager: start the fake daemon, yield its /transcribe url."""

    def __enter__(self):
        _FakeDaemon.seen = []
        self._srv = HTTPServer(("127.0.0.1", 0), _FakeDaemon)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self._srv.server_address[1]}/transcribe"

    def __exit__(self, *a):
        self._srv.shutdown()
