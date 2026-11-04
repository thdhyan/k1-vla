"""E-stop = latch the arbiter, send zero velocity, then ModeFSM.interrupt(): WALKING/CUSTOM/SOCCER -> PREPARE
(stands still in place); DAMPING and PREPARE are left alone, so an e-stop never stands up a limp robot.
Release only unlatches chat motion; it never changes mode (operator goes back to walking with the remote or the
console mode buttons). chat_off latches chat motion without touching the mode.
tests/test_events_safety.py."""
from k1_chat.arbiter import VelocityArbiter
from k1_chat.backend import Loco
from k1_chat.events import EventLog
from k1_chat.modes import ModeFSM


class Safety:
    def __init__(self, loco: Loco, arbiter: VelocityArbiter, log: EventLog, fsm: ModeFSM | None = None):
        self.loco, self.arbiter, self.log = loco, arbiter, log
        self.fsm = fsm or ModeFSM(loco)

    def estop(self, source: str) -> dict:
        self.arbiter.estop()  # first: nothing from chat can move the robot after this line
        out = {"source": source}
        try:
            out["move"] = self.loco.move(0.0, 0.0, 0.0)
            out.update(self.fsm.interrupt())
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        self.log.add("estop", **out)
        return out

    def chat_off(self, source: str) -> dict:
        """Operator takes over: latch chat motion and zero velocity, mode unchanged (no stand-up, no PREPARE)."""
        self.arbiter.estop()
        out = {"source": source, "move": self.loco.move(0.0, 0.0, 0.0)}
        self.log.add("chat_off", **out)
        return out

    def release(self, source: str) -> None:
        self.arbiter.release()
        self.log.add("release", source=source)
