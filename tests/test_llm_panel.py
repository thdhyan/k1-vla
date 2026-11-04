import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from k1_chat.events import EventLog
from k1_chat.llm_panel import ACTIONS, make_panel, summarize


def upstream(reply):
    """Fake vLLM: records requests, answers with `reply`."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            self._send({"data": [{"id": "fake"}]})

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send(reply)

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}", seen


@pytest.fixture
def panel():
    made = []

    def make(routes, runner=None):
        log = EventLog()
        s = make_panel("127.0.0.1", 0, routes, log, runner=runner or (lambda name: iter(["ok"])), status=lambda: {"gpu": "x"})
        threading.Thread(target=s.serve_forever, daemon=True).start()
        made.append(s)
        return f"http://127.0.0.1:{s.server_address[1]}", log

    yield make
    for s in made:
        s.shutdown()


REPLY = {"choices": [{"message": {"role": "assistant", "content": None,
                                  "tool_calls": [{"function": {"name": "turn", "arguments": '{"angle_deg": 90}'}}]}}]}


def test_proxy_forwards_by_model_and_logs_in_and_out(panel):
    up, url, seen = upstream(REPLY)
    base, log = panel({"nemotron-bolt": url})
    body = {"model": "nemotron-bolt", "messages": [{"role": "user", "content": "turn left"}], "tools": [{"x": 1}]}
    r = httpx.post(base + "/v1/chat/completions", json=body, headers={"Authorization": "Bearer local"})
    assert r.json() == REPLY and seen[0]["messages"] == body["messages"]
    kinds = [e["kind"] for e in log.since(0)]
    assert kinds == ["llm_in", "llm_out"]
    out = log.since(0)[1]
    assert out["model"] == "nemotron-bolt" and "turn" in out["text"] and out["ms"] >= 0
    up.shutdown()


def test_unknown_model_is_404_and_logged(panel):
    base, log = panel({})
    r = httpx.post(base + "/v1/chat/completions", json={"model": "nope", "messages": []})
    assert r.status_code == 404 and "not loaded" in r.text


def test_upstream_down_gives_502(panel):
    base, log = panel({"m": "http://127.0.0.1:9"})
    r = httpx.post(base + "/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 502
    assert log.since(0)[-1]["kind"] == "error"


def test_summarize_keeps_text_and_drops_image_bytes():
    jpeg_url = "data:image/jpeg;base64," + "A" * 5000
    s = summarize([{"role": "user", "content": [{"type": "text", "text": "what?"}, {"type": "image_url", "image_url": {"url": jpeg_url}}]}])
    assert "what?" in s and "[image 5000 b64 chars]" in s and "AAAA" not in s


def test_image_in_request_becomes_last_frame(panel):
    import base64
    up, url, _ = upstream({"choices": [{"message": {"content": "a ball"}}]})
    base, log = panel({"cosmos": url})
    jpeg = b"\xff\xd8fakejpeg"
    msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]}]
    httpx.post(base + "/v1/chat/completions", json={"model": "cosmos", "messages": msg})
    assert httpx.get(base + "/api/frame.jpg").content == jpeg
    up.shutdown()


def test_models_lists_routes(panel):
    base, _ = panel({"a": "http://x", "b": "http://y"})
    assert {m["id"] for m in httpx.get(base + "/v1/models").json()["data"]} == {"a", "b"}


def test_page_has_load_unload_buttons(panel):
    base, _ = panel({})
    page = httpx.get(base + "/").text
    for name in ACTIONS:
        assert f'data-a="{name}"' in page


def test_action_runs_whitelisted_script_and_logs_output(panel):
    ran = []
    base, log = panel({}, runner=lambda name: (ran.append(name), iter(["line1", "line2"]))[1])
    r = httpx.post(base + "/api/action/load_vision")
    assert r.status_code == 202
    import time
    for _ in range(50):
        if any(e["kind"] == "action_done" for e in log.since(0)):
            break
        time.sleep(0.02)
    assert ran == ["load_vision"]
    assert [e.get("text") for e in log.since(0) if e["kind"] == "action_out"] == ["line1", "line2"]


def test_unknown_action_rejected_and_get_not_allowed(panel):
    base, _ = panel({})
    assert httpx.post(base + "/api/action/rm_rf").status_code == 404
    assert httpx.get(base + "/api/action/load_chat").status_code == 405


def test_events_api_includes_status(panel):
    base, _ = panel({})
    d = httpx.get(base + "/api/events?since=0").json()
    assert d["status"] == {"gpu": "x"} and d["events"] == []
