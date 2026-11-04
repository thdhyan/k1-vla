"""Client for detect_server (YOLO + depth obstacle, :8092 on the robot). tests/test_perception.py.

get() is None when the service is down or its result is older than MAX_AGE_S: callers treat that as "cannot see",
so forward motion is refused rather than walking blind."""
import json
import urllib.request

MAX_AGE_S = 1.0


class Perception:
    def __init__(self, url: str = "http://127.0.0.1:8092", timeout: float = 0.3, fetch=None):
        self.url, self.timeout = url.rstrip("/"), timeout
        self._fetch = fetch or self._http

    def _http(self, path: str) -> bytes:
        with urllib.request.urlopen(self.url + path, timeout=self.timeout) as r:
            return r.read()

    def get(self) -> dict | None:
        try:
            d = json.loads(self._fetch("/detections"))
        except Exception:  # noqa: BLE001 — down = None
            return None
        age = d.get("age_s")
        return d if age is not None and age <= MAX_AGE_S else None

    def vis(self) -> bytes | None:
        try:
            return self._fetch("/vis.jpg")
        except Exception:  # noqa: BLE001
            return None


def side(bearing_deg: float) -> str:
    if abs(bearing_deg) < 5:
        return "straight ahead"
    return f"{abs(bearing_deg):.0f} degrees {'left' if bearing_deg > 0 else 'right'}"


def describe(d: dict | None, max_objects: int = 8) -> str:
    """One line for the LLM: obstacle ahead plus what YOLO sees, nearest first."""
    if d is None:
        return "camera/depth perception unavailable (cannot check for obstacles)"
    a = d.get("obstacle", {}).get("ahead_m")
    parts = [f"obstacle {a:.1f} m ahead in your walking path" if a is not None else "walking path clear for 3 m"]
    objs = sorted(d.get("objects", []), key=lambda o: o.get("distance_m") or 99)[:max_objects]
    if objs:
        parts.append("detected: " + ", ".join(
            f"{o['tag']} {side(o['bearing_deg'])}" + (f" {o['distance_m']:.1f} m" if o.get("distance_m") else "")
            for o in objs))
    else:
        parts.append("no objects detected")
    return "; ".join(parts)
