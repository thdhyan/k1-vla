"""`look` tool: grab one camera frame, ask the vision model, return its text to the chat model. tests/test_vision.py.

The image goes to a separate VLM call (K1_VISION_MODEL), so the chat history stays text-only and small;
the chat model may be the same model. Every frame sent is shown in the console (EventLog.set_frame).
`watch` sends WATCH_FRAMES frames spread over a few seconds in one VLM call, for motion and what people are doing.
`move_and_watch` runs one motion tool while grabbing a frame every MOVE_INTERVAL_S, then sends them in one VLM call.
"""
import json
import threading
import time

from k1_chat.events import EventLog
from k1_chat.llm import image_part
from k1_chat.tools import Registry, Tool

VISION_SYSTEM = ("You are the eyes of a small humanoid robot; the image is its head camera (about 105 degrees wide, so "
                 "the left and right image edges are about 50 degrees off centre). Answer in one or two short sentences, "
                 "under 30 words, only what was asked. Give directions as degrees left or right of centre and rough "
                 "distance in metres when asked about objects or people.")
WATCH_FRAMES = 4
MOVE_TOOLS = ("turn", "walk", "walk_and_turn")
MOVE_INTERVAL_S, MOVE_FRAMES = 1.0, 8  # 1 fps; a 10 s motion is evenly thinned to 8 frames


def _ask(vlm, log, question: str, images: list[bytes], text: str) -> str:
    log.set_frame(images[-1], prompt=question)
    log.add("frame", prompt=question, bytes=sum(map(len, images)), frames=len(images))
    msg = vlm.complete([{"role": "system", "content": VISION_SYSTEM},
                        {"role": "user", "content": [{"type": "text", "text": text}] + [image_part(j) for j in images]}],
                       max_tokens=1500)
    out = (msg.get("content") or "").strip() or "error: vision model gave no answer"
    log.add("vlm_reply", text=out)
    return out


def add_look_tool(reg: Registry, frames, vlm, log: EventLog, sleep=time.sleep) -> None:
    def look(question: str) -> str:
        jpeg = frames.grab()
        if not jpeg:
            return "error: no camera frame (camera stream unreachable)"
        return _ask(vlm, log, question, [jpeg], question)

    def watch(question: str, seconds: float) -> str:
        seconds = min(max(seconds, 1.0), 6.0)
        dt = seconds / (WATCH_FRAMES - 1)
        shots = []
        for i in range(WATCH_FRAMES):
            if i:
                sleep(dt)
            jpeg = frames.grab()
            if jpeg:
                shots.append(jpeg)
        if not shots:
            return "error: no camera frame (camera stream unreachable)"
        return _ask(vlm, log, question, shots,
                    f"{len(shots)} frames from the head camera, in time order, about {dt:.1f} s apart. {question}")

    def move_and_watch(tool: str, args: dict, question: str) -> str:
        if tool not in MOVE_TOOLS:
            return f"error: tool must be one of {list(MOVE_TOOLS)}"
        shots, done = [], threading.Event()

        def capture():
            while True:
                jpeg = frames.grab()
                if jpeg:
                    shots.append(jpeg)
                if done.wait(MOVE_INTERVAL_S):
                    return

        t = threading.Thread(target=capture, daemon=True)
        t.start()
        try:
            result = reg.run(tool, json.dumps(args or {}))
        finally:
            done.set()
            t.join(timeout=5)
        if result.startswith(("error", "refused", "robot is not")):
            return result
        jpeg = frames.grab()  # where it ended up
        if jpeg:
            shots.append(jpeg)
        if not shots:
            return f"{result}; error: no camera frame (camera stream unreachable)"
        if len(shots) > MOVE_FRAMES:
            shots = [shots[round(i * (len(shots) - 1) / (MOVE_FRAMES - 1))] for i in range(MOVE_FRAMES)]
        seen = _ask(vlm, log, question, shots,
                    f"{len(shots)} frames from the head camera in time order, taken while the robot did {tool} "
                    f"{json.dumps(args or {})} (positive angles = turning left); the last frame is where it stopped. "
                    f"{question}")
        return f"{result}. Seen: {seen}"

    reg.add(Tool("look", "Look through the head camera and answer a question about what is in view.",
                 {"question": {"type": "string", "description": "what to look for or describe"}}, look))
    reg.add(Tool("watch", "Watch through the head camera for a few seconds (several frames) to see movement or what "
                 "someone is doing. Slower than look; use look for a still scene.",
                 {"question": {"type": "string", "description": "what to watch for"},
                  "seconds": {"type": "number", "minimum": 1, "maximum": 6, "description": "how long to watch"}}, watch))
    reg.add(Tool("move_and_watch", "Do one motion (turn, walk or walk_and_turn, same args as that tool) while the head "
                 "camera takes a frame about every second, then answer a question about everything seen on the way. "
                 "Use to scan a room while turning, e.g. tool turn, args {angle_deg: 90}.",
                 {"tool": {"type": "string", "enum": list(MOVE_TOOLS)},
                  "args": {"type": "object", "description": "arguments for that tool"},
                  "question": {"type": "string", "description": "what to look for on the way"}}, move_and_watch))
