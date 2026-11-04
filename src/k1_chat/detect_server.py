"""Perception service on the robot: every YOLO detection + an obstacle check from depth, over HTTP. tests/test_detect_server.py.

Booster's own yolo_detection node is idle until started over RPC and its "get" only returns objects in the centre third
of the image, so k1_chat runs its own: YOLO11n (COCO, onnxruntime CUDA) on /boostercamera/head/rgb (nv12 544x448),
distance from /boostercamera/head/depth (32FC1 millimetres on A1, aligned to the colour image), and /head_pose (camera pose over
the ground under the robot: x forward, y left, z up) so bearings are relative to the body and the floor is not an obstacle.

  robot:  k1voice.sh detect <robot> start|stop      (own process, log ~/k1_chat/detect.log)
  GET /detections  {"t", "age_s", "objects": [{"tag", "conf", "box", "bearing_deg" (+ = left of the body),
                    "distance_m"}], "obstacle": {"ahead_m", "points"}}   ahead_m None = nothing within RANGE_M
  GET /vis.jpg     last frame with boxes          GET /health
"""
import argparse
import ast
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

MODEL = Path.home() / "k1_chat" / "models" / "yolo11n.onnx"
IMG = 640
CONF, IOU = 0.35, 0.5
HALF_WIDTH_M = 0.28              # corridor the robot sweeps when walking straight (K1 ~0.35 m wide + margin)
MIN_H_M, MAX_H_M = 0.06, 1.2     # obstacle height band over the floor
RANGE_M, MIN_POINTS = 3.0, 40
DEPTH_SCALE = 0.001             # 32FC1 depth is in mm (checked on A1)


def letterbox(rgb: np.ndarray, size: int = IMG) -> tuple[np.ndarray, float, tuple[int, int]]:
    import cv2
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    out = np.full((size, size, 3), 114, np.uint8)
    py, px = (size - nh) // 2, (size - nw) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh))
    return out, s, (px, py)


def postprocess(out: np.ndarray, scale: float, pad: tuple[int, int], names: dict, conf: float = CONF,
                iou: float = IOU) -> list[dict]:
    """YOLO11 output (1, 4+classes, N): cx, cy, w, h in letterboxed pixels -> boxes in image pixels after NMS."""
    import cv2
    p = out[0].T
    cls = p[:, 4:].argmax(1)
    score = p[np.arange(len(p)), 4 + cls]
    keep = score >= conf
    p, cls, score = p[keep], cls[keep], score[keep]
    if not len(p):
        return []
    x0 = (p[:, 0] - p[:, 2] / 2 - pad[0]) / scale
    y0 = (p[:, 1] - p[:, 3] / 2 - pad[1]) / scale
    w, h = p[:, 2] / scale, p[:, 3] / scale
    idx = cv2.dnn.NMSBoxesBatched(np.stack([x0, y0, w, h], 1).tolist(), score.tolist(), cls.tolist(), conf, iou)
    return [{"tag": names.get(int(cls[i]), str(int(cls[i]))), "conf": round(float(score[i]), 2),
             "box": [int(x0[i]), int(y0[i]), int(x0[i] + w[i]), int(y0[i] + h[i])]} for i in np.array(idx).flatten()]


def head_yaw(q) -> float:
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rotation(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def to_ground(depth: np.ndarray, K, pos, q, step: int = 4) -> np.ndarray:
    """Depth image (metres, optical frame) -> Nx3 points over the floor under the robot (x fwd, y left, z up)."""
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    d = depth[::step, ::step]
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    ok = np.isfinite(d) & (d > 0.15) & (d < RANGE_M + 1)
    z = d[ok]
    opt = np.stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z], 1)
    body = np.stack([opt[:, 2], -opt[:, 0], -opt[:, 1]], 1)  # optical (right, down, fwd) -> (fwd, left, up)
    return body @ rotation(q).T + np.asarray(pos)


def obstacle(points: np.ndarray) -> dict:
    """Closest thing in the walking corridor that is not floor."""
    c = points[(np.abs(points[:, 1]) < HALF_WIDTH_M) & (points[:, 2] > MIN_H_M) & (points[:, 2] < MAX_H_M)
               & (points[:, 0] > 0.1) & (points[:, 0] < RANGE_M)]
    if len(c) < MIN_POINTS:
        return {"ahead_m": None, "points": int(len(c))}
    return {"ahead_m": round(float(np.percentile(c[:, 0], 3)), 2), "points": int(len(c))}


def locate(obj: dict, depth, K, yaw: float) -> dict:
    """bearing (degrees, + = left of the body) and median depth over the middle of the box."""
    x0, y0, x1, y1 = obj["box"]
    obj["bearing_deg"] = round(math.degrees(math.atan2(K[2] - (x0 + x1) / 2, K[0]) + yaw), 1)
    obj["distance_m"] = None
    if depth is not None:
        mx, my = (x1 - x0) // 4, (y1 - y0) // 4
        patch = depth[y0 + my:y1 - my + 1, x0 + mx:x1 - mx + 1]
        patch = patch[np.isfinite(patch) & (patch > 0.15)]
        if patch.size:
            obj["distance_m"] = round(float(np.median(patch)), 2)
    return obj


def make_server(host: str, port: int, latest, vis) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/detections":
                self._send(json.dumps(latest()).encode(), "application/json")
            elif self.path == "/health":
                d = latest()
                self._send(json.dumps({"ready": d.get("age_s") is not None and d["age_s"] < 2, "age_s": d.get("age_s"),
                                       "hz": d.get("hz")}).encode(), "application/json")
            elif self.path == "/vis.jpg":
                j = vis()
                self._send(j, "image/jpeg") if j else self._send(b"no frame", "text/plain", 503)
            else:
                self._send(b"not found", "text/plain", 404)

    return ThreadingHTTPServer((host, port), H)


class Detector:
    def __init__(self, model: Path = MODEL):
        import onnxruntime as ort
        limit = int(os.environ.get("K1_DETECT_CUDA_MEM_LIMIT", str(256 * 1024 * 1024)))  # unified memory on the Orin
        self.sess = ort.InferenceSession(str(model), providers=[
            ("CUDAExecutionProvider", {"gpu_mem_limit": limit, "arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider"])
        self.names = ast.literal_eval(self.sess.get_modelmeta().custom_metadata_map["names"])
        self.provider = self.sess.get_providers()[0]

    def __call__(self, rgb: np.ndarray) -> list[dict]:
        img, s, pad = letterbox(rgb)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], np.float32) / 255.0
        return postprocess(self.sess.run(None, {"images": x})[0], s, pad, self.names)


class RosPerception:
    """Latest rgb / depth / head pose from ROS; run() detects at `hz` and keeps the newest result."""

    def __init__(self, detector: Detector, hz: float):
        import rclpy
        from geometry_msgs.msg import Pose
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        rclpy.init()
        self.node = rclpy.create_node("k1_chat_detect")
        self.det, self.period = detector, 1.0 / hz
        self.rgb = self.depth = self.K = None
        self.pose = ([0.0, 0.0, 0.89], [0.0, 0.0, 0.0, 1.0])
        self.result = {"objects": [], "obstacle": {"ahead_m": None, "points": 0}, "t": None}
        self.frame = None
        self.lock = threading.Lock()
        sd = qos_profile_sensor_data
        self.node.create_subscription(Image, "/boostercamera/head/rgb", lambda m: setattr(self, "rgb", m), sd)
        self.node.create_subscription(Image, "/boostercamera/head/depth", lambda m: setattr(self, "depth", m), sd)
        self.node.create_subscription(CameraInfo, "/boostercamera/head/rgb/camera_info",
                                      lambda m: setattr(self, "K", list(m.k)), sd)
        self.node.create_subscription(Pose, "/head_pose", self._on_pose, 10)
        ex = SingleThreadedExecutor()
        ex.add_node(self.node)
        threading.Thread(target=ex.spin, daemon=True).start()

    def _on_pose(self, m) -> None:
        self.pose = ([m.position.x, m.position.y, m.position.z],
                     [m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w])

    @staticmethod
    def _rgb(m) -> np.ndarray:
        import cv2
        a = np.frombuffer(bytes(m.data), np.uint8)
        if m.encoding.lower() == "nv12":
            return cv2.cvtColor(a.reshape(m.height * 3 // 2, m.width), cv2.COLOR_YUV2RGB_NV12)
        return a.reshape(m.height, m.width, -1)[..., :3]

    def step(self) -> None:
        m, dm, K = self.rgb, self.depth, self.K
        if m is None or K is None:
            return
        rgb = self._rgb(m)
        depth = np.frombuffer(bytes(dm.data), np.float32).reshape(dm.height, dm.width) * DEPTH_SCALE if dm is not None else None
        pos, q = self.pose
        yaw = head_yaw(q)
        objs = [locate(o, depth, K, yaw) for o in self.det(rgb)]
        obs = obstacle(to_ground(depth, K, pos, q)) if depth is not None else {"ahead_m": None, "points": 0, "no_depth": True}
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            self.result = {"objects": objs, "obstacle": obs, "t": stamp, "head_yaw_deg": round(math.degrees(yaw), 1)}
            self.frame = (rgb, objs, obs)

    def run(self) -> None:
        n, t0 = 0, time.time()
        while True:
            t = time.time()
            try:
                self.step()
                n += 1
            except Exception as e:  # noqa: BLE001 — one bad frame must not end the service
                print(f"[detect] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.hz = round(n / max(time.time() - t0, 1e-6), 1)
            if time.time() - t0 > 10:
                n, t0 = 0, time.time()
            time.sleep(max(0.0, self.period - (time.time() - t)))

    def latest(self) -> dict:
        with self.lock:
            r = dict(self.result)
        r["age_s"] = round(time.time() - r["t"], 2) if r.get("t") else None
        r["hz"] = getattr(self, "hz", None)
        return r

    def vis(self) -> bytes | None:
        import cv2
        with self.lock:
            f = self.frame
        if not f:
            return None
        rgb, objs, obs = f
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for o in objs:
            x0, y0, x1, y1 = o["box"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
            d = f" {o['distance_m']}m" if o.get("distance_m") else ""
            cv2.putText(img, f"{o['tag']} {o['conf']:.2f}{d}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 200, 255), 1)
        a = obs.get("ahead_m")
        cv2.putText(img, f"obstacle ahead: {a} m" if a is not None else "path clear", (6, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if a is not None and a < 0.8 else (0, 255, 0), 1)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--hz", type=float, default=float(os.environ.get("K1_DETECT_HZ", "6")))
    a = ap.parse_args(argv)
    det = Detector()
    p = RosPerception(det, a.hz)
    import signal
    for sig in (signal.SIGTERM, signal.SIGINT):  # rclpy.init took these over; it only stops the executor
        signal.signal(sig, lambda *_: os._exit(0))
    threading.Thread(target=p.run, daemon=True).start()
    print(f"detect_server on {a.host}:{a.port} ({det.provider}, {a.hz} Hz)", flush=True)
    make_server(a.host, a.port, p.latest, p.vis).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
