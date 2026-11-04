"""Robot backend: one call(api_id, param) seam, mirroring B1LocoClient::SendApiRequest. tests/test_backend.py.

API ids and JSON fields come from booster_ws/sdk/booster_robotics_sdk/include/booster/robot/b1/b1_loco_api.hpp
and b1_loco_client.hpp. SimRobot is the virtual robot; the real transport (ROS /LocoApiTopicReq or SDK python)
lives in booster_robot.py and only has to implement call().
"""
import math
from enum import IntEnum
from typing import Protocol


class Mode(IntEnum):  # common/robot_shared.hpp RobotMode
    DAMPING = 0
    PREPARE = 1
    WALKING = 2
    CUSTOM = 3
    SOCCER = 4


class Api(IntEnum):  # b1_loco_api.hpp LocoApiId
    CHANGE_MODE = 2000
    MOVE = 2001
    ROTATE_HEAD = 2004
    WAVE_HAND = 2005
    HANDSHAKE = 2015
    DANCE = 2016
    GET_MODE = 2017
    GET_STATUS = 2018


HAND_OPEN, HAND_CLOSE = 0, 1  # b1_api_const.hpp HandAction
RIGHT_HAND = 1                # b1_api_const.hpp HandIndex, WaveHand always uses it
DANCE_ID = {"dab": 3, "cheer": 6}  # b1_loco_api.hpp DanceId kDabbingGesture, kCheeringGesture
DANCE_STOP = 1000                  # DanceId kStop: dances never end by themselves (checked on A2)
ERR_BUSY = 501                     # real K1 answer when another action is still in GetStatus current_actions


def encode(api: Api, **param) -> tuple[int, dict]:
    return int(api), {k: int(v) if isinstance(v, IntEnum) else v for k, v in param.items()}


class RobotBackend(Protocol):
    def call(self, api_id: int, param: dict) -> tuple[int, dict]:
        """Send one loco API request; returns (code, response body). code 0 = ok."""


class Loco:
    """Typed wrapper over any backend (same method names as B1LocoClient, snake_case)."""

    def __init__(self, backend: RobotBackend):
        self.backend = backend

    def _send(self, api: Api, **param) -> int:
        return self.backend.call(*encode(api, **param))[0]

    def move(self, vx: float, vy: float, vyaw: float) -> int:
        return self._send(Api.MOVE, vx=vx, vy=vy, vyaw=vyaw)

    def rotate_head(self, pitch: float, yaw: float) -> int:
        return self._send(Api.ROTATE_HEAD, pitch=pitch, yaw=yaw)

    def wave_hand(self, open_: bool) -> int:
        return self._send(Api.WAVE_HAND, hand_index=RIGHT_HAND, hand_action=HAND_OPEN if open_ else HAND_CLOSE)

    def handshake(self, open_: bool) -> int:
        return self._send(Api.HANDSHAKE, hand_action=HAND_OPEN if open_ else HAND_CLOSE)

    def dance(self, dance_id: int) -> int:
        return self._send(Api.DANCE, dance_id=dance_id)

    def change_mode(self, mode: Mode) -> int:
        return self._send(Api.CHANGE_MODE, mode=mode)

    def current_actions(self) -> list[int] | None:
        """GetStatus current_actions (robot_shared.hpp Action ids); None if unreadable."""
        code, resp = self.backend.call(int(Api.GET_STATUS), {})
        return list(resp["current_actions"]) if code == 0 and "current_actions" in resp else None

    def get_mode(self) -> Mode | None:
        code, resp = self.backend.call(int(Api.GET_MODE), {})
        return Mode(resp["mode"]) if code == 0 and "mode" in resp else None


ERR_MODE = -2  # sim only: request not valid in the current mode (real error codes: verify on robot)
STANDING = (Mode.PREPARE, Mode.WALKING)


class SimRobot:
    """Virtual K1: mode rules, last twist, head angles, planar pose, and a log of every call."""

    def __init__(self, mode: Mode = Mode.PREPARE):
        self.mode = mode
        self.twist = (0, 0, 0)
        self.head = (0.0, 0.0)
        self.pose = [0.0, 0.0, 0.0]  # x, y, yaw in the start frame
        self.calls: list[tuple[Api, dict]] = []
        self.actions: list[int] = []        # GetStatus current_actions
        self.settle_s = 0.0                 # how long an action stays listed after it is ended
        self._ending: dict[int, float] = {}
        self.rejected = 0

    def call(self, api_id: int, param: dict) -> tuple[int, dict]:
        api = Api(api_id)
        self.calls.append((api, dict(param)))
        if api == Api.GET_MODE:
            return 0, {"mode": int(self.mode)}
        if api == Api.GET_STATUS:
            return 0, {"current_actions": list(self.actions), "current_mode": int(self.mode)}
        if api == Api.CHANGE_MODE:
            new = Mode(param["mode"])
            if new == Mode.WALKING and self.mode not in STANDING:
                return ERR_MODE, {}
            self.mode, self.twist = new, (0, 0, 0)
            return 0, {}
        if api == Api.MOVE:
            if self.mode != Mode.WALKING:
                return ERR_MODE, {}
            self.twist = (param["vx"], param["vy"], param["vyaw"])
            return 0, {}
        if api in (Api.WAVE_HAND, Api.HANDSHAKE, Api.DANCE) and self.mode != Mode.WALKING:
            return ERR_MODE, {}  # real K1: gestures only in WALKING
        if self.mode not in STANDING:
            return ERR_MODE, {}
        if api == Api.ROTATE_HEAD:
            self.head = (param["pitch"], param["yaw"])
        if api in (Api.WAVE_HAND, Api.HANDSHAKE, Api.DANCE):
            return self._action(api, param)
        return 0, {}

    def _action(self, api: Api, param: dict) -> tuple[int, dict]:
        # robot_shared.hpp Action: 1 handshake, 2 wave, 7 dab, 10 cheer
        start = param.get("hand_action") == HAND_OPEN if api != Api.DANCE else param["dance_id"] != DANCE_STOP
        if start:
            if self.actions:
                self.rejected += 1
                return ERR_BUSY, {}
            self.actions.append({Api.HANDSHAKE: 1, Api.WAVE_HAND: 2}.get(api) or {3: 7, 6: 10}[param["dance_id"]])
            return 0, {}
        for a in list(self.actions):
            self._end(a)
        return 0, {}

    def _end(self, action: int) -> None:
        if self.settle_s <= 0:
            self.actions.remove(action)
        else:
            self._ending.setdefault(action, self.settle_s)

    def advance(self, dt: float) -> None:
        for a in list(self._ending):
            self._ending[a] -= dt
            if self._ending[a] <= 1e-9:
                del self._ending[a]
                self.actions.remove(a)
        vx, vy, w = self.twist
        x, y, th = self.pose
        self.pose = [x + (vx * math.cos(th) - vy * math.sin(th)) * dt,
                     y + (vx * math.sin(th) + vy * math.cos(th)) * dt, th + w * dt]
