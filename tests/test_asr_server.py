import base64
import threading

import httpx

from k1_chat.asr_server import make_server
from k1_chat.robot_audio import NemotronAsr


def test_nemotron_client_talks_to_laptop_server():
    got = []
    srv = make_server("127.0.0.1", 0, lambda pcm, lang: got.append((len(pcm), lang)) or "turn left", {"model": "fake"})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        asr = NemotronAsr(url=url + "/transcribe")
        assert asr.health() and asr(b"\0\0" * 1600) == "turn left" and got == [(3200, "en")]
        bad = httpx.post(url + "/transcribe", json={"pcm16": base64.b64encode(b"\0\0").decode(), "sr": 8000})
        assert bad.status_code == 400
    finally:
        srv.shutdown()
