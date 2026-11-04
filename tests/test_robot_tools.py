import json, math

from k1_chat.arbiter import VelocityArbiter
from k1_chat.backend import Api, Loco, Mode, SimRobot
from k1_chat.events import EventLog
from k1_chat.robot_tools import build_registry
from k1_chat.safety import Safety


class Clock:
    def __init__(self, sim):
        self.t, self.sim = 0.0, sim

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.sim.advance(dt)
        self.t += dt


def make(mode=Mode.WALKING, on_sleep=None):
    sim = SimRobot(mode=mode)
    clk = Clock(sim)
    arb = VelocityArbiter(clock=clk)
    log = EventLog()
    loco = Loco(sim)
    safety = Safety(loco, arb, log)

    def sleep(dt):
        clk.sleep(dt)
        if on_sleep:
            on_sleep(clk.t, safety)

    reg = build_registry(loco, arb, log, sleep=sleep, clock=clk)
    return reg, sim, arb


def names(reg):
    return sorted(t["function"]["name"] for t in reg.schemas())


def test_tool_set():
    reg, *_ = make()
    assert names(reg) == ["gesture", "handshake", "robot_status", "set_head", "set_mode", "stop", "turn", "walk", "walk_and_turn", "wave"]


def test_walk_forward_one_metre_then_zero():
    reg, sim, _ = make()
    out = reg.run("walk", json.dumps({"distance_m": 1.0, "speed": "normal"}))
    assert "walked" in out
    assert abs(sim.pose[0] - 1.0) < 0.05 and abs(sim.pose[1]) < 1e-6
    assert sim.twist == (0, 0, 0)                     # always ends with a zero Move
    assert sim.calls[-1] == (Api.MOVE, {"vx": 0.0, "vy": 0.0, "vyaw": 0.0})


def test_walk_backwards_and_distance_is_clamped():
    reg, sim, _ = make()
    reg.run("walk", json.dumps({"distance_m": -9, "speed": "slow"}))
    assert -0.55 < sim.pose[0] < -0.45                # clamped to -0.5 m


def test_every_move_respects_arbiter_limits():
    reg, sim, _ = make()
    reg.run("walk_and_turn", json.dumps({"vx": 5, "vyaw": 5, "duration_s": 1}))
    moves = [p for a, p in sim.calls if a == Api.MOVE]
    assert max(abs(p["vx"]) for p in moves) <= 0.4 and max(abs(p["vyaw"]) for p in moves) <= 0.5


def test_turn_left_90_degrees():
    reg, sim, _ = make()
    reg.run("turn", json.dumps({"angle_deg": 90}))
    assert abs(math.degrees(sim.pose[2]) - 90) < 3


def test_motion_refused_when_not_walking_mode():
    reg, sim, _ = make(mode=Mode.PREPARE)
    out = reg.run("walk", json.dumps({"distance_m": 0.5, "speed": "normal"}))
    assert "walking mode" in out and sim.pose == [0.0, 0.0, 0.0]
    assert all(p.get("mode") != int(Mode.WALKING) for a, p in sim.calls if a == Api.CHANGE_MODE)  # never switches mode itself


def test_estop_mid_walk_aborts():
    reg, sim, arb = make(on_sleep=lambda t, s: s.estop("console") if t > 0.5 and not s.arbiter.estopped else None)
    out = reg.run("walk", json.dumps({"distance_m": 1.5, "speed": "normal"}))
    assert "stopped" in out and sim.pose[0] < 0.3 and sim.mode == Mode.PREPARE


def test_head_in_degrees_clamped():
    reg, sim, _ = make()
    reg.run("set_head", json.dumps({"yaw_deg": 90, "pitch_deg": 10}))
    assert sim.head == (math.radians(10), math.radians(45))


def test_wave_opens_then_closes():
    reg, sim, _ = make()
    reg.run("wave", json.dumps({"seconds": 3}))
    acts = [p["hand_action"] for a, p in sim.calls if a == Api.WAVE_HAND]
    assert acts == [0, 1]


def test_handshake_opens_then_closes():
    reg, sim, _ = make()
    reg.run("handshake", json.dumps({"seconds": 4}))
    assert [p["hand_action"] for a, p in sim.calls if a == Api.HANDSHAKE] == [0, 1]


def test_stop_zeroes_without_mode_change():
    reg, sim, _ = make()
    sim.twist = (0.3, 0, 0)
    reg.run("stop", "{}")
    assert sim.twist == (0, 0, 0) and sim.mode == Mode.WALKING


def test_tools_refused_while_estopped():
    reg, sim, arb = make()
    arb.estop()
    assert "e-stop" in reg.run("wave", json.dumps({"seconds": 2}))
    assert not any(a == Api.WAVE_HAND for a, _ in sim.calls)


def test_set_mode_prepare_from_damping():
    reg, sim, _ = make(mode=Mode.DAMPING)
    out = reg.run("set_mode", json.dumps({"mode": "prepare"}))
    assert sim.mode == Mode.PREPARE and "prepare" in out


def test_set_mode_walking_needs_prepare_first():
    reg, sim, _ = make(mode=Mode.DAMPING)
    out = reg.run("set_mode", json.dumps({"mode": "walking"}))
    assert sim.mode == Mode.DAMPING and "error" in out


def test_set_mode_never_damping():
    reg, sim, _ = make(mode=Mode.WALKING)
    reg.run("set_mode", json.dumps({"mode": "damping"}))
    assert sim.mode == Mode.WALKING


def test_set_mode_refused_while_estopped():
    reg, sim, arb = make(mode=Mode.DAMPING)
    arb.estop()
    assert "e-stop" in reg.run("set_mode", json.dumps({"mode": "prepare"}))
    assert sim.mode == Mode.DAMPING


def test_gesture_dab_and_cheer_send_dance_ids():
    reg, sim, _ = make(mode=Mode.WALKING)
    assert "dab done" in reg.run("gesture", json.dumps({"name": "dab"}))
    assert "cheer done" in reg.run("gesture", json.dumps({"name": "cheer"}))
    assert [p["dance_id"] for a, p in sim.calls if a == Api.DANCE] == [3, 1000, 6, 1000]


def test_gesture_refused_when_lying_down():
    reg, sim, _ = make(mode=Mode.DAMPING)
    assert "walking mode" in reg.run("gesture", json.dumps({"name": "dab"}))


def test_gestures_refused_in_prepare():
    reg, sim, _ = make(mode=Mode.PREPARE)
    for name, args in [("wave", {"seconds": 2}), ("handshake", {"seconds": 2}), ("gesture", {"name": "dab"})]:
        assert "walking mode" in reg.run(name, json.dumps(args))
    assert not [a for a, _ in sim.calls if a in (Api.WAVE_HAND, Api.HANDSHAKE, Api.DANCE)]


def actions_log(sim):
    return [(a, p) for a, p in sim.calls if a in (Api.WAVE_HAND, Api.HANDSHAKE, Api.DANCE)]


def test_dance_is_stopped_and_robot_left_idle():
    reg, sim, _ = make(mode=Mode.WALKING)
    reg.run("gesture", json.dumps({"name": "cheer"}))
    assert [p["dance_id"] for _, p in actions_log(sim)] == [6, 1000] and sim.actions == []


def test_gestures_back_to_back_wait_for_idle():
    reg, sim, _ = make(mode=Mode.WALKING)
    sim.settle_s = 1.5  # real K1: the arm needs ~2 s after close before GetStatus is empty again
    for name, args in [("wave", {"seconds": 2}), ("handshake", {"seconds": 2}), ("gesture", {"name": "dab"})]:
        assert "done" in reg.run(name, json.dumps(args))
    assert sim.rejected == 0


def test_gesture_refused_if_robot_stays_busy():
    reg, sim, _ = make(mode=Mode.WALKING)
    sim.actions = [12]  # something else (e.g. the remote) keeps an action running
    assert "busy" in reg.run("wave", json.dumps({"seconds": 2}))
    assert actions_log(sim) == []


def test_robot_status_reports_mode_motion_actions_battery():
    sim = SimRobot(mode=Mode.WALKING)
    arb = VelocityArbiter()
    reg = build_registry(Loco(sim), arb, sleep=lambda dt: None,
                         battery=lambda: {"soc": 86.6, "voltage": 51.8, "current": -1.8})
    out = reg.run("robot_status", "{}")
    assert "walking" in out and "chat motion on" in out and "battery 87%" in out and "no gesture running" in out
    arb.estop()
    sim.actions = [2]
    out = build_registry(Loco(sim), arb, sleep=lambda dt: None).run("robot_status", "{}")
    assert "chat motion off" in out and "battery unknown" in out and "wave" in out
