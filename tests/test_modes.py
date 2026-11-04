import threading

from k1_chat.backend import Api, Loco, Mode, SimRobot
from k1_chat.modes import ModeFSM


def make(mode):
    sim = SimRobot(mode=mode)
    return ModeFSM(Loco(sim), sleep=lambda dt: None), sim


def changes(sim):
    return [Mode(p["mode"]) for a, p in sim.calls if a == Api.CHANGE_MODE]


def test_allowed_edges():
    fsm, sim = make(Mode.DAMPING)
    assert fsm.request(Mode.PREPARE) is None and sim.mode == Mode.PREPARE
    assert fsm.request(Mode.WALKING) is None and sim.mode == Mode.WALKING
    assert fsm.request(Mode.PREPARE) is None and sim.mode == Mode.PREPARE


def test_never_damping():
    for start in (Mode.PREPARE, Mode.WALKING):
        fsm, sim = make(start)
        assert "not allowed" in fsm.request(Mode.DAMPING)
        assert changes(sim) == []


def test_no_skipping_damping_to_walking():
    fsm, sim = make(Mode.DAMPING)
    assert "prepare first" in fsm.request(Mode.WALKING)
    assert changes(sim) == []


def test_hands_off_custom_and_soccer():
    for start in (Mode.CUSTOM, Mode.SOCCER):
        fsm, sim = make(start)
        assert "error" in fsm.request(Mode.PREPARE) and changes(sim) == []


def test_same_mode_is_noop():
    fsm, sim = make(Mode.PREPARE)
    assert fsm.request(Mode.PREPARE) is None and changes(sim) == []


def test_mode_change_refused_while_action_runs():
    fsm, sim = make(Mode.PREPARE)
    with fsm.action() as ok:
        assert ok
        assert "busy" in fsm.request(Mode.WALKING)
    assert fsm.request(Mode.WALKING) is None


def test_second_action_refused_while_first_runs():
    fsm, _ = make(Mode.WALKING)
    with fsm.action():
        with fsm.action() as ok:
            assert not ok


def test_unreadable_mode_refuses_change():
    class Dead:
        def call(self, api_id, param):
            return -1, {"error": "timeout"}
    fsm = ModeFSM(Loco(Dead()), sleep=lambda dt: None)
    assert "cannot read" in fsm.request(Mode.PREPARE)


def test_waits_until_robot_reports_target():
    class Slow(SimRobot):  # real K1 takes a while to stand up: GetMode reports the old mode for a bit
        lag = 0
        def call(self, api_id, param):
            if api_id == int(Api.CHANGE_MODE):
                self.lag = 3
            elif api_id == int(Api.GET_MODE) and self.lag:
                self.lag -= 1
                return 0, {"mode": int(Mode.DAMPING)}
            return super().call(api_id, param)
    sim = Slow(mode=Mode.DAMPING)
    fsm = ModeFSM(Loco(sim), sleep=lambda dt: None)
    assert fsm.request(Mode.PREPARE) is None
    assert sim.lag == 0


def test_interrupt_walking_goes_prepare():
    for start in (Mode.WALKING, Mode.CUSTOM, Mode.SOCCER):
        fsm, sim = make(start)
        fsm.interrupt()
        assert sim.mode == Mode.PREPARE


def test_interrupt_never_stands_up_or_damps():
    for start in (Mode.DAMPING, Mode.PREPARE):
        fsm, sim = make(start)
        fsm.interrupt()
        assert sim.mode == start and changes(sim) == []


def test_interrupt_does_not_wait_for_running_action():
    fsm, sim = make(Mode.WALKING)
    done = threading.Event()
    with fsm.action():
        threading.Thread(target=lambda: (fsm.interrupt(), done.set())).start()
        assert done.wait(2)
    assert sim.mode == Mode.PREPARE
