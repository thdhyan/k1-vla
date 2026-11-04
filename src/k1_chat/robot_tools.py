"""Robot tools for the LLM: narrow flat tools over one Loco backend. tests/test_robot_tools.py.

Motion goes through the VelocityArbiter (chat clamped, estop latches). A motion tool blocks for its duration,
re-sending Move at 10 Hz, and always ends with one zero Move; between tool calls we send nothing, so the
physical remote keeps full control. Open loop: distances and angles are approximate.
Motion tools refuse when the robot is not in walking mode. With a Perception (detect_server), forward motion stops
before an obstacle in the walking corridor and is refused while depth is unavailable; follow_person steers on YOLO. Mode changes go only through ModeFSM (modes.py):
one allowed edge at a time, never DAMPING, never while a motion or gesture runs (each holds fsm.action()).
"""
import math
import time

from k1_chat.arbiter import Twist, VelocityArbiter
from k1_chat.backend import DANCE_ID, DANCE_STOP, Loco, Mode
from k1_chat.events import EventLog
from k1_chat.modes import ModeFSM
from k1_chat.tools import Registry, Tool

HZ = 10
SPEED = {"slow": 0.15, "normal": 0.3}  # m/s, under the arbiter's 0.4 cap
YAW_RATE = 0.4                         # rad/s
MAX_S = 10.0                           # longest single motion
DANCE_S = 4.0                          # dab/cheer hold before Dance stop (they never end by themselves)
IDLE_WAIT_S = 6.0                      # wait for GetStatus current_actions to empty (arm needs ~2 s after close)
NOT_WALKING = ("not moved: robot is in {mode} mode, not in walking mode. Call set_mode walking first (from prepare), "
               "then retry. Do not say you moved.")
STOP_M = 0.5                           # stop forward motion when an obstacle is this close in the walking corridor
NO_DEPTH = "not moved: cannot see obstacles (perception service down or stale), so forward walking is refused"
FOLLOW_GAIN, FOLLOW_LOST_S = 1.2, 2.0  # yaw rate per rad of bearing; stop after the person is gone this long
ESTOPPED = "refused: e-stop is active (operator must release it in the console)"
GESTURE_NEEDS_WALKING = "refused: gestures need walking mode; ask to switch to walking mode first (set_mode walking)."
ROBOT_BUSY = "refused: the robot is still busy with another gesture; try again in a moment."
ACTION_NAMES = {1: "handshake", 2: "wave", 7: "dab", 10: "cheer", 12: "boxing"}  # robot_shared.hpp Action
BUSY = "refused: another action or mode change is running; wait or say stop"


def build_registry(loco: Loco, arbiter: VelocityArbiter, log: EventLog | None = None,
                   sleep=time.sleep, clock=time.monotonic, fsm: ModeFSM | None = None, battery=None,
                   perception=None) -> Registry:
    reg = Registry(log)
    fsm = fsm or ModeFSM(loco, sleep=sleep)

    def exclusive(fn):  # one motion/gesture at a time, and never during a mode change
        def run(*a, **kw):
            with fsm.action() as ok:
                return fn(*a, **kw) if ok else BUSY
        return run

    def wait_idle() -> bool:  # K1 rejects (400/501) or overlaps a gesture sent while another is still listed
        t0 = clock()
        while True:
            acts = loco.current_actions()
            if not acts:  # [] idle; None = status unreadable, do not block on it
                return True
            if clock() - t0 >= IDLE_WAIT_S:
                return False
            sleep(0.25)

    def not_walking() -> str | None:
        mode = loco.get_mode()  # K1 accepts Move in PREPARE and does nothing, so check first
        return NOT_WALKING.format(mode=mode.name.lower()) if mode is not None and mode != Mode.WALKING else None

    def blocked(t0: float) -> str | None:  # forward motion only; None = go on
        if perception is None:
            return None
        d = perception.get()
        if d is None:
            return NO_DEPTH if clock() - t0 < 1e-9 else f"stopped after {clock() - t0:.1f} s: {NO_DEPTH[11:]}"
        a = d.get("obstacle", {}).get("ahead_m")
        if a is not None and a < STOP_M:
            return (f"not moved: obstacle {a:.1f} m ahead" if clock() - t0 < 1e-9 else
                    f"stopped after {clock() - t0:.1f} s: obstacle {a:.1f} m ahead")
        return None

    @exclusive
    def drive(twist: Twist, seconds: float, done: str) -> str:
        if arbiter.estopped:
            return ESTOPPED
        if err := not_walking():
            return err
        seconds = min(seconds, MAX_S)
        t0 = clock()
        try:
            while clock() - t0 < seconds - 1e-9:
                if arbiter.estopped:
                    return "stopped by e-stop"
                if twist.vx > 0 and (err := blocked(t0)):
                    return err
                arbiter.chat(twist)
                out = arbiter.output()
                if loco.move(out.vx, out.vy, out.vyaw) != 0:
                    return NOT_WALKING.format(mode="a non-walking")
                sleep(1 / HZ)
            return done
        finally:
            arbiter.chat(Twist())
            loco.move(0.0, 0.0, 0.0)

    def walk(distance_m: float, speed: str) -> str:
        v = SPEED[speed]
        return drive(Twist(vx=math.copysign(v, distance_m)), abs(distance_m) / v, f"walked about {distance_m:.2f} m")

    def turn(angle_deg: float) -> str:
        a = math.radians(angle_deg)
        return drive(Twist(vyaw=math.copysign(YAW_RATE, a)), abs(a) / YAW_RATE, f"turned about {angle_deg:.0f} degrees")

    def walk_and_turn(vx: float, vyaw: float, duration_s: float) -> str:
        return drive(Twist(vx=vx, vyaw=vyaw), duration_s, f"moved for {duration_s:.1f} s")

    def stop() -> str:
        arbiter.chat(Twist())
        loco.move(0.0, 0.0, 0.0)
        return "stopped"

    @exclusive
    def set_head(yaw_deg: float, pitch_deg: float) -> str:
        if arbiter.estopped:
            return ESTOPPED
        code = loco.rotate_head(math.radians(pitch_deg), math.radians(yaw_deg))
        return f"head at yaw {yaw_deg:.0f}, pitch {pitch_deg:.0f}" if code == 0 else f"error: head command failed (code {code})"

    def gesture(send, label):
        @exclusive
        def run(seconds: float) -> str:
            if arbiter.estopped:
                return ESTOPPED
            if loco.get_mode() != Mode.WALKING:  # K1: hand gestures only run in WALKING (checked on A2)
                return GESTURE_NEEDS_WALKING
            if not wait_idle():
                return ROBOT_BUSY
            code = send(True)
            if code != 0:
                return f"error: {label} failed (code {code})"
            sleep(seconds)
            send(False)
            wait_idle()
            return f"{label} done"
        return run

    def set_mode(mode: str) -> str:
        if arbiter.estopped:
            return ESTOPPED
        target = {"prepare": Mode.PREPARE, "walking": Mode.WALKING}.get(mode)
        if target is None:
            return f"error: unknown mode {mode!r}; only prepare or walking"
        arbiter.chat(Twist())
        err = fsm.request(target)
        return err or f"now in {mode} mode"

    @exclusive
    def dance_gesture(name: str) -> str:
        if arbiter.estopped:
            return ESTOPPED
        if loco.get_mode() != Mode.WALKING:
            return GESTURE_NEEDS_WALKING
        if not wait_idle():
            return ROBOT_BUSY
        code = loco.dance(DANCE_ID[name])
        if code != 0:
            return f"error: {name} failed (code {code})"
        sleep(DANCE_S)
        loco.dance(DANCE_STOP)
        wait_idle()
        return f"{name} done"

    def robot_status() -> str:  # read-only
        mode = loco.get_mode()
        acts = loco.current_actions()
        b = battery() if battery else None
        parts = [f"mode {mode.name.lower() if mode is not None else 'unknown'}",
                 "chat motion off (e-stop latched in the console)" if arbiter.estopped else "chat motion on",
                 "gesture status unknown" if acts is None else
                 (f"running: {', '.join(ACTION_NAMES.get(a, str(a)) for a in acts)}" if acts else "no gesture running"),
                 f"battery {b['soc']:.0f}% ({b['voltage']:.1f} V, {b['current']:+.1f} A)" if b else "battery unknown"]
        return "; ".join(parts)

    @exclusive
    def follow_person(seconds: float, distance_m: float) -> str:
        from k1_chat.perception import side
        if arbiter.estopped:
            return ESTOPPED
        if err := not_walking():
            return err
        t0 = last_seen = clock()
        target, seconds = None, min(max(seconds, 2.0), 60.0)
        try:
            while clock() - t0 < seconds:
                if arbiter.estopped:
                    return "stopped by e-stop"
                d = perception.get()
                if d is None:
                    return f"stopped after {clock() - t0:.0f} s: {NO_DEPTH[11:]}"
                people = [o for o in d.get("objects", []) if o["tag"] == "person" and o["conf"] >= 0.4]
                if people:  # keep the same person: nearest in bearing to the last one, else the biggest box
                    target = (min(people, key=lambda o: abs(o["bearing_deg"] - target["bearing_deg"])) if target else
                              max(people, key=lambda o: (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])))
                    last_seen = clock()
                    b = math.radians(target["bearing_deg"])
                    vyaw = max(-0.5, min(0.5, FOLLOW_GAIN * b))
                    dist = target.get("distance_m")
                    a = d.get("obstacle", {}).get("ahead_m")
                    vx = 0.0
                    if dist and abs(target["bearing_deg"]) < 25 and (a is None or a >= STOP_M):
                        vx = max(0.0, min(0.3, 0.6 * (dist - distance_m)))
                    twist = Twist(vx=vx, vyaw=vyaw)
                elif clock() - last_seen > FOLLOW_LOST_S:
                    return f"lost the person after {clock() - t0:.0f} s and stopped" if target else "no person in view, did not move"
                else:
                    twist = Twist()
                arbiter.chat(twist)
                out = arbiter.output()
                if loco.move(out.vx, out.vy, out.vyaw) != 0:
                    return NOT_WALKING.format(mode="a non-walking")
                sleep(1 / HZ)
            if not target:
                return "no person in view, did not move"
            dist = f", about {target['distance_m']:.1f} m away" if target.get("distance_m") else ""
            return f"followed for {seconds:.0f} s; person last {side(target['bearing_deg'])}{dist}"
        finally:
            arbiter.chat(Twist())
            loco.move(0.0, 0.0, 0.0)

    def detect() -> str:
        from k1_chat.perception import describe
        return describe(perception.get())

    num = lambda lo, hi, desc: {"type": "number", "minimum": lo, "maximum": hi, "description": desc}
    reg.add(Tool("walk", "Walk straight forward (positive) or backward (negative). Blocks until done.",
                 {"distance_m": num(-0.5, 1.5, "metres, negative = backward"),
                  "speed": {"type": "string", "enum": list(SPEED)}}, walk))
    reg.add(Tool("turn", "Turn in place. Positive = left (counter-clockwise), negative = right.",
                 {"angle_deg": num(-180, 180, "degrees")}, turn))
    reg.add(Tool("walk_and_turn", "Walk along an arc: forward speed plus turn rate for a duration.",
                 {"vx": num(-0.2, 0.4, "m/s forward"), "vyaw": num(-0.5, 0.5, "rad/s, positive = left"),
                  "duration_s": num(0.5, 5, "seconds")}, walk_and_turn))
    reg.add(Tool("stop", "Stop walking now.", {}, stop))
    reg.add(Tool("set_head", "Turn only the head (neck), not the body, e.g. to look left/right/up/down before turning. "
                 "yaw: positive = left. pitch: positive = down. yaw 0, pitch 0 = straight ahead.",
                 {"yaw_deg": num(-45, 45, "degrees"), "pitch_deg": num(-25, 25, "degrees")}, set_head))
    reg.add(Tool("wave", "Wave the right hand at someone. Needs walking mode.", {"seconds": num(1, 5, "how long to wave")},
                 gesture(loco.wave_hand, "wave")))
    reg.add(Tool("handshake", "Hold out the hand for a handshake, then pull it back. Needs walking mode.",
                 {"seconds": num(2, 10, "how long to hold the hand out")}, gesture(loco.handshake, "handshake")))
    reg.add(Tool("gesture", "Do a short upper-body gesture: dab or cheer. Needs walking mode.",
                 {"name": {"type": "string", "enum": list(DANCE_ID)}}, dance_gesture))
    reg.add(Tool("set_mode", "Change locomotion mode. prepare = stand up stiff in place (from lying limp/damping); "
                 "walking = ready to walk, only from prepare. Only when the person asks for it.",
                 {"mode": {"type": "string", "enum": ["prepare", "walking"]}}, set_mode))
    reg.add(Tool("robot_status", "Read your own state: locomotion mode, whether chat motion is allowed, running "
                 "gesture, battery. Read-only; use it when asked how you are, your battery, or which mode you are in.",
                 {}, robot_status))
    if perception is not None:
        reg.add(Tool("detect", "Fast object detection (YOLO + depth) on the head camera: every object with direction "
                     "(degrees left/right of your body) and distance, plus the obstacle distance in your walking path. "
                     "Use it to turn toward something by the exact angle. No colours or details: use look for those.",
                     {}, detect))
        reg.add(Tool("follow_person", "Follow the person in front of you: turn to keep them centred and walk to stay "
                     "about distance_m away, stopping for obstacles. Blocks for `seconds` or until the person is lost.",
                     {"seconds": num(2, 60, "how long to follow"),
                      "distance_m": num(0.8, 2.5, "distance to keep, metres")}, follow_person))
    return reg
