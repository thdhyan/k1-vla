"""Thread-safe ring buffer of brain events (heard, llm_request, llm_reply, tool_call, tool_result, said, estop, ...)
for the web console, plus the latest camera frame sent to the vision model. tests/test_events_safety.py."""
import threading
import time
from collections import deque


class EventLog:
    def __init__(self, maxlen: int = 500):
        self._ev: deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0
        self._frame: tuple[bytes, dict] | None = None

    def add(self, kind: str, **data) -> int:
        with self._lock:
            self._seq += 1
            self._ev.append({"seq": self._seq, "t": time.time(), "kind": kind, **data})
            return self._seq

    def since(self, seq: int) -> list[dict]:
        with self._lock:
            return [e for e in self._ev if e["seq"] > seq]

    def set_frame(self, jpeg: bytes, **meta) -> None:
        with self._lock:
            self._frame = (jpeg, {"t": time.time(), **meta})

    def frame(self) -> tuple[bytes, dict] | None:
        with self._lock:
            return self._frame
