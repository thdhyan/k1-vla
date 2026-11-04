"""Camera frame sources (JPEG bytes). tests/test_vision.py.

StudioFrames: the robot's Studio websocket ws://<ip>:51111 pushes field1=10001 messages, payload 3 { 1: JPEG }
(RGB, ~10 fps, see booster_ws/docs/booster_websocket_studio_cli_findings_2026-10-27.md). No ssh or ROS needed,
works from the laptop or on the robot (127.0.0.1). We connect per grab: the stream is ~10 Mbit/s.
FileFrames: a fixture image for tests and laptop runs without a robot.
"""
from pathlib import Path

CAMERA = 10001


def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return v, i


def _fields(b: bytes) -> dict[int, list]:
    out: dict[int, list] = {}
    i = 0
    while i < len(b):
        tag, i = _varint(b, i)
        f, w = tag >> 3, tag & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        elif w == 1:
            v, i = b[i:i + 8], i + 8
        elif w == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {w}")
        out.setdefault(f, []).append(v)
    return out


def extract_jpeg(msg: bytes) -> bytes | None:
    try:
        top = _fields(msg)
        if top.get(1) != [CAMERA]:
            return None
        jpeg = _fields(top[3][0])[1][0]
    except (ValueError, IndexError, KeyError, TypeError):
        return None
    return jpeg if jpeg[:2] == b"\xff\xd8" else None


class StudioFrames:
    def __init__(self, ip: str, port: int = 51111, timeout: float = 3.0, connect=None):
        self.url, self.timeout = f"ws://{ip}:{port}", timeout
        if connect is None:
            from websockets.sync.client import connect
        self._connect = connect

    def grab(self) -> bytes | None:
        try:
            with self._connect(self.url, open_timeout=self.timeout, max_size=None) as ws:
                for _ in range(60):  # status messages are interleaved, ~1 Hz
                    m = ws.recv(timeout=self.timeout)
                    if isinstance(m, bytes) and (j := extract_jpeg(m)):
                        return j
        except Exception:
            return None
        return None


class FileFrames:
    def __init__(self, path: Path):
        self.path = Path(path)

    def grab(self) -> bytes | None:
        return self.path.read_bytes()
