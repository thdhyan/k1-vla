import threading

import httpx, pytest
from k1_chat.arbiter import VelocityArbiter
from k1_chat.backend import Loco, Mode, SimRobot
from k1_chat.console import make_server
from k1_chat.events import EventLog
from k1_chat.safety import Safety


@pytest.fixture
def console():
    sim, arb, log = SimRobot(mode=Mode.WALKING), VelocityArbiter(), EventLog()
    safety = Safety(Loco(sim), arb, log)
    srv = make_server("127.0.0.1", 0, "A2", log, safety, state=lambda: {"mode": sim.mode.name})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", sim, arb, log
    srv.shutdown()


def test_page_names_robot_and_has_stop_button(console):
    url, *_ = console
    html = httpx.get(url + "/").text
    assert "A2" in html and "STOP" in html


def test_estop_post_puts_robot_in_prepare(console):
    url, sim, arb, _ = console
    r = httpx.post(url + "/api/estop")
    assert r.status_code == 200 and r.json()["change_mode"] == 0
    assert sim.mode == Mode.PREPARE and arb.estopped


def test_estop_get_is_rejected(console):  # link prefetch / crawler must not stop the robot
    url, sim, *_ = console
    assert httpx.get(url + "/api/estop").status_code == 405 and sim.mode == Mode.WALKING


def test_release_post(console):
    url, _, arb, _ = console
    httpx.post(url + "/api/estop"); httpx.post(url + "/api/release")
    assert not arb.estopped


def test_events_since_and_state(console):
    url, _, _, log = console
    a = log.add("heard", text="hello")
    log.add("said", text="hi")
    body = httpx.get(url + f"/api/events?since={a}").json()
    assert [e["kind"] for e in body["events"]] == ["said"]
    assert body["state"]["mode"] == "WALKING" and body["state"]["estopped"] is False


def test_frame_endpoint(console):
    url, _, _, log = console
    assert httpx.get(url + "/api/frame.jpg").status_code == 404
    log.set_frame(b"\xff\xd8x", prompt="p")
    r = httpx.get(url + "/api/frame.jpg")
    assert r.status_code == 200 and r.content == b"\xff\xd8x" and r.headers["content-type"] == "image/jpeg"


def test_live_camera_frame():
    sim, arb, log = SimRobot(mode=Mode.WALKING), VelocityArbiter(), EventLog()
    srv = make_server("127.0.0.1", 0, "A2", log, Safety(Loco(sim), arb, log), live=lambda: b"\xff\xd8jpeg")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = httpx.get(f"http://127.0.0.1:{srv.server_address[1]}/api/live.jpg")
        assert r.status_code == 200 and r.content == b"\xff\xd8jpeg" and r.headers["content-type"] == "image/jpeg"
    finally:
        srv.shutdown()


def test_live_camera_absent_is_404(console):
    url, *_ = console
    assert httpx.get(url + "/api/live.jpg").status_code == 404


def test_chat_motion_off_latches_without_mode_change(console):
    url, sim, arb, _ = console
    assert httpx.post(url + "/api/chat_off").status_code == 200
    assert arb.estopped and sim.mode == Mode.WALKING
    httpx.post(url + "/api/chat_on")
    assert not arb.estopped and sim.mode == Mode.WALKING


def test_mode_buttons_go_through_fsm():
    sim, arb, log = SimRobot(mode=Mode.DAMPING), VelocityArbiter(), EventLog()
    srv = make_server("127.0.0.1", 0, "A2", log, Safety(Loco(sim), arb, log))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert "prepare first" in httpx.post(url + "/api/mode/walking").json()["result"]
        assert httpx.post(url + "/api/mode/prepare").json()["result"] == "now in prepare mode" and sim.mode == Mode.PREPARE
        assert httpx.post(url + "/api/mode/walking").json()["result"] == "now in walking mode" and sim.mode == Mode.WALKING
        assert httpx.post(url + "/api/mode/damping").status_code == 404 and sim.mode == Mode.WALKING
        assert httpx.get(url + "/api/mode/prepare").status_code == 405
    finally:
        srv.shutdown()


def test_debug_endpoint():
    sim, arb, log = SimRobot(mode=Mode.WALKING), VelocityArbiter(), EventLog()
    srv = make_server("127.0.0.1", 0, "A2", log, Safety(Loco(sim), arb, log), debug=lambda: {"memory": {"available_mb": 1}})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = httpx.get(f"http://127.0.0.1:{srv.server_address[1]}/api/debug")
        assert r.status_code == 200 and r.json() == {"memory": {"available_mb": 1}}
    finally:
        srv.shutdown()


def test_debug_absent_is_404(console):
    url, *_ = console
    assert httpx.get(url + "/api/debug").status_code == 404 and "System" in httpx.get(url + "/").text


def test_calibrate_button():
    from k1_chat.audio import LiveCalibration, Segmenter
    sim, arb, log = SimRobot(mode=Mode.WALKING), VelocityArbiter(), EventLog()
    cal = LiveCalibration(Segmenter(16000, 30, start_rms=300.0), 30, lambda t: None)
    srv = make_server("127.0.0.1", 0, "A2", log, Safety(Loco(sim), arb, log), calibrate=cal)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert httpx.get(url + "/api/calibrate").json()["phase"] == "idle"
        assert httpx.post(url + "/api/calibrate").json()["phase"] == "quiet"
        assert "calibrate mic" in httpx.get(url + "/").text and sim.mode == Mode.WALKING
    finally:
        srv.shutdown()


def test_calibrate_absent_is_404(console):
    url, *_ = console
    assert httpx.post(url + "/api/calibrate").status_code == 404 and httpx.get(url + "/api/calibrate").status_code == 404
