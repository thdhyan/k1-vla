from pathlib import Path

import pytest
from k1_chat.events import EventLog
from k1_chat.frames import FileFrames, StudioFrames, extract_jpeg
from k1_chat.tools import Registry, Tool
from k1_chat.vision import add_look_tool

FIX = Path(__file__).parent / "fixtures" / "ball_right.jpg"


def varint(n):
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def ld(field, payload):  # length-delimited protobuf field
    return varint(field << 3 | 2) + varint(len(payload)) + payload


def studio_msg(kind, jpeg=b""):
    # docs: field1 = 10001 camera frame, payload 3 { 1: JPEG }
    return varint(1 << 3) + varint(kind) + ld(3, ld(1, jpeg))


def test_extract_jpeg_from_studio_camera_message():
    assert extract_jpeg(studio_msg(10001, b"\xff\xd8abc")) == b"\xff\xd8abc"
    assert extract_jpeg(studio_msg(10000, b"status")) is None
    assert extract_jpeg(b"\x00\x01garbage") is None


def test_studio_frames_skips_status_messages():
    class FakeWS:
        def __init__(self): self.msgs = [studio_msg(10000, b"s"), "text", studio_msg(10001, b"\xff\xd8jpg")]
        def recv(self, timeout=None): return self.msgs.pop(0)
        def __enter__(self): return self
        def __exit__(self, *a): pass
    assert StudioFrames("1.2.3.4", connect=lambda url, **kw: FakeWS()).grab() == b"\xff\xd8jpg"


def test_studio_frames_returns_none_when_robot_unreachable():
    def boom(url, **kw): raise OSError("no route")
    assert StudioFrames("1.2.3.4", connect=boom).grab() is None


def test_file_frames():
    assert FileFrames(FIX).grab()[:2] == b"\xff\xd8"


class FakeVLM:
    def __init__(self, text): self.text, self.calls = text, []
    def complete(self, messages, **kw):
        self.calls.append(messages)
        return {"content": self.text}


def test_look_sends_frame_and_question_and_logs_for_console():
    log, reg, vlm = EventLog(), Registry(), FakeVLM("A red ball on the right.")
    add_look_tool(reg, FileFrames(FIX), vlm, log)
    out = reg.run("look", '{"question": "where is the ball?"}')
    assert out == "A red ball on the right."
    user = vlm.calls[0][-1]["content"]
    assert user[0]["text"] == "where is the ball?" and user[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert log.frame()[0] == FIX.read_bytes()
    assert [e["kind"] for e in log.since(0)] == ["frame", "vlm_reply"]


def test_look_without_camera_says_so():
    class NoCam:
        def grab(self): return None
    reg = Registry()
    add_look_tool(reg, NoCam(), FakeVLM("x"), EventLog())
    assert "no camera frame" in reg.run("look", '{"question": "hi"}')


@pytest.mark.net
def test_live_vlm_sees_ball_on_the_right():
    from dotenv import load_dotenv; load_dotenv()
    from k1_chat.llm import from_env
    reg = Registry()
    add_look_tool(reg, FileFrames(FIX), from_env("K1_VISION_MODEL"), EventLog())
    out = reg.run("look", '{"question": "Is there a ball? Answer left, right or centre in one word."}').lower()
    assert "right" in out


@pytest.mark.robot
def test_live_studio_websocket_frame():
    import os
    j = StudioFrames(os.environ["K1_IP"]).grab()
    assert j and j[:2] == b"\xff\xd8"


def test_watch_sends_several_frames_in_time_order():
    class Seq:
        n = 0
        def grab(self):
            self.n += 1
            return b"\xff\xd8frame%d" % self.n
    log, reg, vlm, slept = EventLog(), Registry(), FakeVLM("A person is waving."), []
    add_look_tool(reg, Seq(), vlm, log, sleep=slept.append)
    assert reg.run("watch", '{"question": "what is the person doing?", "seconds": 3}') == "A person is waving."
    user = vlm.calls[0][-1]["content"]
    images = [p for p in user if p["type"] == "image_url"]
    assert len(images) == 4 and "in time order" in user[0]["text"] and "what is the person doing?" in user[0]["text"]
    assert slept == [1.0, 1.0, 1.0]
    assert log.frame()[0] == b"\xff\xd8frame4"


def _seq_frames():
    class Seq:
        n = 0
        def grab(self):
            self.n += 1
            return b"\xff\xd8frame%d" % self.n
    return Seq()


def test_move_and_watch_grabs_frames_during_the_motion(monkeypatch):
    import time as _time
    from k1_chat import vision
    monkeypatch.setattr(vision, "MOVE_INTERVAL_S", 0.02)
    log, reg, vlm, ran = EventLog(), Registry(), FakeVLM("A person on the left."), []

    def turn(angle_deg: float) -> str:
        ran.append(angle_deg)
        _time.sleep(0.2)
        return "turned about 90 degrees"
    reg.add(Tool("turn", "turn", {"angle_deg": {"type": "number"}}, turn))
    add_look_tool(reg, _seq_frames(), vlm, log)
    out = reg.run("move_and_watch", '{"tool": "turn", "args": {"angle_deg": 90}, "question": "any person?"}')
    assert out == "turned about 90 degrees. Seen: A person on the left."
    assert ran == [90]
    user = vlm.calls[0][-1]["content"]
    images = [p for p in user if p["type"] == "image_url"]
    assert 3 <= len(images) <= vision.MOVE_FRAMES and "while the robot did turn" in user[0]["text"]


def test_move_and_watch_skips_vision_when_motion_refused_and_rejects_other_tools():
    log, reg, vlm = EventLog(), Registry(), FakeVLM("x")
    reg.add(Tool("turn", "turn", {"angle_deg": {"type": "number"}}, lambda angle_deg: "refused: e-stop is active"))
    add_look_tool(reg, _seq_frames(), vlm, log)
    assert reg.run("move_and_watch", '{"tool": "turn", "args": {"angle_deg": 90}, "question": "q"}').startswith("refused")
    assert reg.run("move_and_watch", '{"tool": "wave", "args": {}, "question": "q"}').startswith("error")
    assert vlm.calls == []
