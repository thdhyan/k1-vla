"""Run ON a K1 (humble). Serves RGB | colorized depth side by side as MJPEG.

  source /opt/ros/humble/setup.bash; python3 depth_sbs_server.py [--port 8090]
  view: http://<robot-ip>:8090/   (stream: /stream.mjpg, one frame: /snapshot.jpg, stats: /stats)
"""
import argparse, socket, threading, time
import cv2, numpy as np, rclpy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from rclpy.node import Node
from sensor_msgs.msg import Image

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--max-m", type=float, default=6.0, help="depth range shown (m); topic is float32 millimetres")
ap.add_argument("--quality", type=int, default=75)
args = ap.parse_args()

cond = threading.Condition()
state = {"jpg": b"", "n": 0, "t0": time.time(), "rgb": None, "rgb_n": 0, "depth_n": 0}


class Sbs(Node):
    def __init__(self):
        super().__init__("depth_sbs")
        self.create_subscription(Image, "/boostercamera/head/rgb", self.on_rgb, 5)
        self.create_subscription(Image, "/boostercamera/head/depth", self.on_depth, 5)

    def on_rgb(self, m):
        buf = np.frombuffer(m.data, np.uint8)
        if m.encoding == "nv12":
            img = cv2.cvtColor(buf.reshape(m.height * 3 // 2, m.width), cv2.COLOR_YUV2BGR_NV12)
        else:
            img = buf.reshape(m.height, m.width, -1)[:, :, :3]
        state["rgb"] = img
        state["rgb_n"] += 1

    def on_depth(self, m):
        state["depth_n"] += 1
        d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        d8 = np.clip(np.nan_to_num(d) / (args.max_m * 1000) * 255, 0, 255).astype(np.uint8)
        col = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
        col[d <= 0] = 0
        rgb = state["rgb"]
        if rgb is None:
            return
        if rgb.shape[:2] != col.shape[:2]:
            rgb = cv2.resize(rgb, (col.shape[1], col.shape[0]))
        ok, j = cv2.imencode(".jpg", np.hstack([rgb, col]), [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if ok:
            with cond:
                state["jpg"] = j.tobytes()
                state["n"] += 1
                cond.notify_all()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/stats":
            dt = time.time() - state["t0"]
            body = f'{{"sbs_fps":{state["n"]/dt:.1f},"rgb_fps":{state["rgb_n"]/dt:.1f},"depth_fps":{state["depth_n"]/dt:.1f}}}'.encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path.startswith("/snapshot"):
            with cond:
                j = state["jpg"]
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(j))); self.end_headers(); self.wfile.write(j)
        elif self.path.startswith("/stream"):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(3)  # drop stale tabs instead of writing to them forever
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            last = 0
            try:
                while True:
                    with cond:
                        if not cond.wait_for(lambda: state["n"] != last, 5):
                            continue
                        last, j = state["n"], state["jpg"]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
            except OSError:  # BrokenPipe, ConnectionReset, timeout
                pass
        else:
            b = b'<body style="margin:0;background:#111"><img src="/stream.mjpg" style="width:100%"></body>'
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)


rclpy.init()
threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever, daemon=True).start()
rclpy.spin(Sbs())
