"""Real K1 backend: loco RPC over ROS 2 (/LocoApiTopicReq -> /LocoApiTopicResp), the same DDS topics
B1LocoClient uses. Runs ON the robot (humble; source /opt/ros/humble and /opt/booster/ros2 first).
Wire format from sdk/.../robot/rpc/request_header.hpp and response_header.hpp. tests/test_booster_robot.py.

UNVERIFIED until checked on a robot (see PLAN_TOOLS.md "morning checklist"): message type name (discovered at
runtime), response matching by uuid, and status codes.
"""
import json
import threading
import uuid as uuidlib

REQ, RESP = "/LocoApiTopicReq", "/LocoApiTopicResp"


def encode_request(api_id: int, param: dict, uuid: str) -> dict:
    return {"uuid": uuid, "header": json.dumps({"api_id": api_id, "expect_response": True}),
            "body": json.dumps(param) if param else ""}


def decode_response(header: str, body: str) -> tuple[int, dict]:
    try:
        status = int(json.loads(header)["status"])
    except (ValueError, KeyError, TypeError):
        return -1, {"error": f"bad response header {header[:80]!r}"}
    try:
        return status, json.loads(body) if body else {}
    except ValueError:
        return status, {"raw": body}


def _spin(executor) -> None:
    from rclpy.executors import ExternalShutdownException
    try:
        executor.spin()
    except ExternalShutdownException:  # SIGTERM shuts rclpy down under the thread: normal stop, not an error
        pass


class RosLocoBackend:
    def __init__(self, timeout: float = 2.0, node_name: str = "k1_chat_loco"):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rosidl_runtime_py.utilities import get_message

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        types = dict(self._wait_types([REQ, RESP]))
        self._Req, Resp = get_message(types[REQ]), get_message(types[RESP])
        self._pub = self._node.create_publisher(self._Req, REQ, 10)
        self._pending: dict[str, tuple[threading.Event, list]] = {}
        self._node.create_subscription(Resp, RESP, self._on_resp, 10)
        self._timeout = timeout
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._spin = threading.Thread(target=_spin, args=(self._exec,), daemon=True)
        self._spin.start()
        self._wait_matched()

    def _wait_matched(self, tries: int = 50):
        # DDS discovery takes ~1 s on the K1 (checked on A2): a request published before the loco service
        # subscribes is dropped and times out, so the first call after startup always failed.
        import time
        for _ in range(tries):
            if self._pub.get_subscription_count() > 0:
                return
            time.sleep(0.1)
        raise RuntimeError(f"no subscriber on {REQ}: is the robot's loco service running?")

    def _wait_types(self, topics, tries: int = 30):
        import time
        for _ in range(tries):
            known = {n: t[0] for n, t in self._node.get_topic_names_and_types()}
            if all(t in known for t in topics):
                return [(t, known[t]) for t in topics]
            time.sleep(0.1)
        raise RuntimeError(f"topics {topics} not visible: is ROS sourced and ROS_DOMAIN_ID=0?")

    def _on_resp(self, msg) -> None:
        waiter = self._pending.get(msg.uuid)
        if waiter:
            waiter[1].append(decode_response(msg.header, msg.body))
            waiter[0].set()

    def call(self, api_id: int, param: dict) -> tuple[int, dict]:
        uid = str(uuidlib.uuid4())
        ev, box = threading.Event(), []
        self._pending[uid] = (ev, box)
        try:
            self._pub.publish(self._Req(**encode_request(api_id, param, uid)))
            if not ev.wait(self._timeout):
                return -1, {"error": "timeout"}
            return box[0]
        finally:
            self._pending.pop(uid, None)

    def close(self) -> None:
        import rclpy
        self._exec.shutdown()
        self._spin.join(timeout=2)
        self._node.destroy_node()
        rclpy.try_shutdown()  # without it the spin thread aborts the process at exit ("terminate called ...")


class RosFrames:
    """Head camera JPEGs from /booster_video_stream (sensor_msgs/CompressedImage, jpeg, 10 Hz on A2).
    Same grab() as frames.StudioFrames; needs rclpy, so robot only."""

    def __init__(self, topic: str = "/booster_video_stream", timeout: float = 2.0, max_age: float = 1.0):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("k1_chat_frames")
        self._latest: tuple[float, bytes] | None = None
        self._got = threading.Event()
        self._node.create_subscription(CompressedImage, topic, self._on_img, qos_profile_sensor_data)
        self._timeout, self._max_age = timeout, max_age
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._spin = threading.Thread(target=_spin, args=(self._exec,), daemon=True)
        self._spin.start()

    def _on_img(self, msg) -> None:
        # age from the image stamp, not arrival: a stalled camera keeps republishing its last frame at 10 Hz (seen on A2)
        self._latest = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, bytes(msg.data))
        self._got.set()

    def grab(self) -> bytes | None:
        import time
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            latest = self._latest
            if latest and abs(time.time() - latest[0]) <= self._max_age:
                return latest[1]
            self._got.clear()
            self._got.wait(max(0.0, deadline - time.monotonic()))
        return None

    def close(self) -> None:
        self._exec.shutdown()
        self._spin.join(timeout=2)
        self._node.destroy_node()


class RosBattery:
    """/battery_state (booster_interface/msg/BatteryState: voltage, current, soc, average_voltage). The type ships only
    inside a vendor agent bundle that robot/run.sh sources when present; without it, or with no message for 10 s,
    read() is None."""

    def __init__(self, topic: str = "/battery_state", max_age: float = 10.0):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rosidl_runtime_py.utilities import get_message

        Msg = get_message("booster_interface/msg/BatteryState")  # raises if the bundle is not sourced
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("k1_chat_battery")
        self._latest: tuple[float, dict] | None = None
        self._node.create_subscription(Msg, topic, self._on_msg, 10)
        self._max_age = max_age
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._spin = threading.Thread(target=_spin, args=(self._exec,), daemon=True)
        self._spin.start()

    def _on_msg(self, msg) -> None:
        import time
        self._latest = (time.monotonic(), {"soc": msg.soc, "voltage": msg.voltage, "current": msg.current})

    def read(self) -> dict | None:
        import time
        latest = self._latest
        return latest[1] if latest and time.monotonic() - latest[0] <= self._max_age else None


def main(argv=None) -> int:
    """On the robot, ROS sourced:  python -m k1_chat.booster_robot get-mode | prepare | frame [out.jpg]
    get-mode and frame are read-only. prepare = fallback e-stop when the brain is dead: zero Move, then
    ModeFSM.interrupt() (WALKING/CUSTOM/SOCCER -> PREPARE; DAMPING and PREPARE are left alone)."""
    import sys
    from k1_chat.backend import Loco
    from k1_chat.modes import ModeFSM
    args = argv or sys.argv[1:] or ["get-mode"]
    cmd = args[0]
    if cmd == "frame":
        f = RosFrames()
        try:
            jpeg = f.grab()
            print("frame bytes:", len(jpeg) if jpeg else None, flush=True)
            if jpeg and len(args) > 1:
                open(args[1], "wb").write(jpeg)
            return 0 if jpeg else 1
        finally:
            f.close()
            import rclpy
            rclpy.try_shutdown()
    b = RosLocoBackend()
    try:
        loco = Loco(b)
        if cmd == "prepare":
            print("move 0:", loco.move(0.0, 0.0, 0.0), "interrupt:", ModeFSM(loco).interrupt(), flush=True)
        mode = loco.get_mode()
        print("mode:", mode.name if mode is not None else "unknown", flush=True)
        return 0 if mode is not None else 1
    finally:
        b.close()


if __name__ == "__main__":
    raise SystemExit(main())
