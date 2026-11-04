"""Velocity arbiter: manual control always beats chat. tests/test_arbiter.py defines the contract."""
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Twist:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


class VelocityArbiter:
    def __init__(self, max_vx=0.4, max_vy=0.2, max_vyaw=0.5, chat_timeout=1.0, manual_hold=2.0, clock=None):
        self._lim = (max_vx, max_vy, max_vyaw)
        self._chat_timeout, self._manual_hold = chat_timeout, manual_hold
        self._clock = clock or time.monotonic
        self._manual: Twist | None = None
        self._manual_t = float("-inf")
        self._chat = Twist()
        self._chat_t = float("-inf")
        self._estop = False

    def manual(self, t: Twist) -> None:
        self._manual, self._manual_t = t, self._clock()
        self._chat = Twist()

    def chat(self, t: Twist) -> None:
        now = self._clock()
        if self._estop or now - self._manual_t < self._manual_hold:
            return
        self._chat = Twist(_clip(t.vx, self._lim[0]), _clip(t.vy, self._lim[1]), _clip(t.vyaw, self._lim[2]))
        self._chat_t = now

    def estop(self) -> None:
        self._estop = True
        self._chat = Twist()

    def release(self) -> None:
        self._estop = False

    @property
    def estopped(self) -> bool:
        return self._estop

    def output(self) -> Twist:
        now = self._clock()
        if self._estop:
            return Twist()
        if self._manual is not None and now - self._manual_t < self._manual_hold:
            return self._manual
        if now - self._chat_t < self._chat_timeout:
            return self._chat
        return Twist()
