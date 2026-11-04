"""Locomotion mode FSM shared by the chat tools and the e-stop. tests/test_modes.py.

Only these edges, one step at a time, each confirmed by GetMode before the next is allowed:
  DAMPING -> PREPARE (stand up), PREPARE -> WALKING, WALKING -> PREPARE.
Nothing ever goes to DAMPING (it drops the robot). CUSTOM/SOCCER belong to other controllers: hands off,
except the interrupt. Mode changes are refused while a motion/gesture holds action().
interrupt() (stop / e-stop): WALKING/CUSTOM/SOCCER -> PREPARE; DAMPING and PREPARE are left as they are,
so an interrupt never makes a limp robot stand up and never damps a standing one. It does not wait for
the running action (that action sees the arbiter latch and ends itself).
"""
import threading
import time
from contextlib import contextmanager

from k1_chat.backend import Loco, Mode

EDGES = {(Mode.DAMPING, Mode.PREPARE), (Mode.PREPARE, Mode.WALKING), (Mode.WALKING, Mode.PREPARE)}
SETTLE_S = 6.0  # longest wait for GetMode to report the new mode (standing up takes a few seconds)


class ModeFSM:
    def __init__(self, loco: Loco, sleep=time.sleep, poll_s: float = 0.2):
        self.loco, self._sleep, self._poll = loco, sleep, poll_s
        self._busy = threading.Lock()   # held by a running motion/gesture or mode change

    @contextmanager
    def action(self):
        ok = self._busy.acquire(blocking=False)
        try:
            yield ok
        finally:
            if ok:
                self._busy.release()

    def request(self, target: Mode) -> str | None:
        """None on success, else an error string for the LLM."""
        if target == Mode.DAMPING:
            return "error: damping is not allowed from chat (it would drop the robot)"
        if not self._busy.acquire(blocking=False):
            return "error: busy with another action; say stop first"
        try:
            cur = self.loco.get_mode()
            if cur is None:
                return "error: cannot read the robot's mode, not changing it"
            if cur == target:
                return None
            if (cur, target) not in EDGES:
                if cur == Mode.DAMPING and target == Mode.WALKING:
                    return "error: the robot is limp; go to prepare first, then walking"
                return f"error: cannot go from {cur.name.lower()} to {target.name.lower()} from chat"
            return self._change(target)
        finally:
            self._busy.release()

    def interrupt(self) -> dict:
        cur = self.loco.get_mode()
        if cur in (None, Mode.DAMPING, Mode.PREPARE):
            return {"mode": cur.name if cur is not None else "unknown", "change_mode": "skipped"}
        return {"mode": cur.name, "change_mode": self.loco.change_mode(Mode.PREPARE)}

    def _change(self, target: Mode) -> str | None:
        code = self.loco.change_mode(target)
        if code != 0:
            return f"error: mode change to {target.name.lower()} failed (code {code})"
        for _ in range(int(SETTLE_S / self._poll)):
            if self.loco.get_mode() == target:
                return None
            self._sleep(self._poll)
        return f"error: robot did not report {target.name.lower()} mode after {SETTLE_S:.0f} s"
