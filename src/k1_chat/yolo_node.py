#!/usr/bin/env python3
"""YOLO on the K1's left ZED eye: ``/{ns}/zed/left/image_raw`` ->
``/{ns}/yolo/image_raw`` (annotated rgb8, same stamp; the headset's YOLO panel
and the sim's debug video show it) + ``/{ns}/yolo/detections`` (JSON String:
class, score, xyxy px). Drops frames it cannot keep up with (latest only).

Needs rclpy + ultralytics + torch. From the Isaac Lab venv (no system ROS) it
re-execs with Isaac Sim's bundled rclpy:

    source scripts/phase6_env.sh
    $PHASE6_VENV/bin/python -m k1_teleop.yolo_node --weights yolov8n.pt
"""
from __future__ import annotations

import argparse
import json
import threading
import time

try:
    import rclpy
except ImportError:                      # Isaac Lab venv: use the bundled Jazzy rclpy
    from k1_utils import isaac_ros_env

    isaac_ros_env.ensure()
    import rclpy

import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ns", default="k1_0")
    ap.add_argument("--weights", default="yolov8n.pt", help="ultralytics weights (downloaded on first use)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0", help="'cpu' or a CUDA index")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(a.weights)
    rclpy.init()
    node = rclpy.create_node("k1_yolo")
    big = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
    pub_img = node.create_publisher(Image, f"/{a.robot_ns}/yolo/image_raw", big)
    pub_det = node.create_publisher(String, f"/{a.robot_ns}/yolo/detections", 10)
    latest: dict = {}
    node.create_subscription(Image, f"/{a.robot_ns}/zed/left/image_raw", lambda m: latest.__setitem__("m", m), big)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    node.get_logger().info(f"YOLO {a.weights} on /{a.robot_ns}/zed/left/image_raw")

    n, t_log, ms = 0, time.time(), 0.0
    while rclpy.ok():
        m = latest.pop("m", None)
        if m is None:
            time.sleep(0.003)
            continue
        if m.encoding != "rgb8":
            continue
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        t0 = time.perf_counter()
        r = model.predict(rgb[..., ::-1], conf=a.conf, device=a.device, verbose=False)[0]   # ultralytics wants BGR
        ms = 1e3 * (time.perf_counter() - t0)
        vis = np.ascontiguousarray(rgb)
        dets = []
        for box, cls, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = (int(v) for v in box)
            name = r.names[int(cls)]
            dets.append({"class": name, "score": round(float(score), 3), "xyxy": [x0, y0, x1, y1]})
            color = (255, 60, 220) if name == "person" else (60, 220, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(vis, f"{name} {score:.2f}", (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(vis, f"YOLO {len(dets)} det  {ms:.0f} ms", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        out = Image(height=m.height, width=m.width, encoding="rgb8", step=m.width * 3)
        out.header = m.header
        out.data = vis.tobytes()
        pub_img.publish(out)
        pub_det.publish(String(data=json.dumps({"stamp": [m.header.stamp.sec, m.header.stamp.nanosec], "dets": dets})))
        n += 1
        if time.time() - t_log > 5.0:
            node.get_logger().info(f"{n / (time.time() - t_log):.1f} Hz, {ms:.0f} ms/frame, {len(dets)} det")
            n, t_log = 0, time.time()


if __name__ == "__main__":
    main()
