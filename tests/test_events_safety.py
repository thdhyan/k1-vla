import threading

from k1_chat.arbiter import Twist, VelocityArbiter
from k1_chat.backend import Api, Loco, Mode, SimRobot
from k1_chat.events import EventLog
from k1_chat.safety import Safety


def test_eventlog_since_returns_newer_events_in_order():
    log = EventLog(maxlen=3)
    for i in range(5):
        log.add("x", i=i)
    ev = log.since(0)
    assert [e["i"] for e in ev] == [2, 3, 4]          # ring buffer keeps the newest
    assert [e["i"] for e in log.since(ev[1]["seq"])] == [4]


def test_eventlog_keeps_latest_frame_separately():
    log = EventLog()
    log.set_frame(b"jpg1", prompt="what do you see")
    assert log.frame()[0] == b"jpg1"


def test_eventlog_is_thread_safe():
    log = EventLog(maxlen=10000)
    ts = [threading.Thread(target=lambda: [log.add("x") for _ in range(500)]) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    seqs = [e["seq"] for e in log.since(0)]
    assert len(seqs) == 2000 and len(set(seqs)) == 2000


def test_estop_zeroes_motion_goes_to_prepare_and_latches():
    sim, arb, log = SimRobot(mode=Mode.WALKING), VelocityArbiter(), EventLog()
    s = Safety(Loco(sim), arb, log)
    arb.chat(Twist(vx=0.3))
    s.estop("console")
    assert sim.mode == Mode.PREPARE and sim.twist == (0, 0, 0)
    assert [c[0] for c in sim.calls][-3:] == [Api.MOVE, Api.GET_MODE, Api.CHANGE_MODE]
    assert arb.output() == Twist() and arb.estopped
    arb.chat(Twist(vx=0.3)); assert arb.output() == Twist()
    assert log.since(0)[-1]["kind"] == "estop" and log.since(0)[-1]["source"] == "console"


def test_estop_from_damping_never_stands_up():
    sim, arb = SimRobot(mode=Mode.DAMPING), VelocityArbiter()
    Safety(Loco(sim), arb, EventLog()).estop("voice")
    assert sim.mode == Mode.DAMPING and Api.CHANGE_MODE not in [c[0] for c in sim.calls] and arb.estopped


def test_release_unlatches_but_never_changes_mode():
    sim, arb = SimRobot(mode=Mode.WALKING), VelocityArbiter()
    s = Safety(Loco(sim), arb, EventLog())
    s.estop("voice"); n = len(sim.calls)
    s.release("console")
    assert not arb.estopped and sim.mode == Mode.PREPARE and len(sim.calls) == n


def test_estop_still_latches_when_backend_raises():
    class Dead:
        def call(self, *a):
            raise OSError("robot gone")
    arb, log = VelocityArbiter(), EventLog()
    Safety(Loco(Dead()), arb, log).estop("console")
    assert arb.estopped and "OSError" in log.since(0)[-1]["error"]
