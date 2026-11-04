import json, os

import pytest
from k1_chat.backend import Api, Loco
from k1_chat.booster_robot import decode_response, encode_request


def test_request_matches_sdk_rpc_wire_format():
    # rpc/request_header.hpp: {"api_id", "expect_response"}; body = the *Parameter ToJson dump
    f = encode_request(int(Api.MOVE), {"vx": 0.2, "vy": 0.0, "vyaw": 0.0}, uuid="u1")
    assert f["uuid"] == "u1"
    assert json.loads(f["header"]) == {"api_id": 2001, "expect_response": True}
    assert json.loads(f["body"]) == {"vx": 0.2, "vy": 0.0, "vyaw": 0.0}


def test_empty_param_is_empty_body_like_sdk():  # LieDown/GetMode send ""
    assert encode_request(int(Api.GET_MODE), {}, uuid="u")["body"] == ""


def test_response_status_and_body():
    # rpc/response_header.hpp: {"status": int}
    assert decode_response('{"status": 0}', '{"mode": 2}') == (0, {"mode": 2})
    assert decode_response('{"status": 400}', "") == (400, {})
    assert decode_response("garbage", "") [0] != 0


@pytest.mark.robot
def test_live_get_mode_only_reads():
    """On the robot: source /opt/ros/humble + /opt/booster/ros2, then pytest -m robot. Read-only."""
    from k1_chat.booster_robot import RosLocoBackend
    b = RosLocoBackend()
    try:
        assert Loco(b).get_mode() is not None
    finally:
        b.close()
