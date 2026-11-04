import json

from k1_chat.backend import Api, Loco, Mode, SimRobot, encode


def test_encode_matches_sdk_json():
    # b1_loco_client.hpp: Move -> MoveParameter{vx,vy,vyaw}, RotateHead{pitch,yaw}, WaveHand{hand_index,hand_action}
    assert encode(Api.MOVE, vx=0.2, vy=0.0, vyaw=0.1) == (2001, {"vx": 0.2, "vy": 0.0, "vyaw": 0.1})
    assert encode(Api.CHANGE_MODE, mode=Mode.PREPARE) == (2000, {"mode": 1})
    assert json.loads(json.dumps(encode(Api.WAVE_HAND, hand_index=1, hand_action=0)[1])) == {"hand_index": 1, "hand_action": 0}


def test_sim_rejects_move_unless_walking():
    r = SimRobot(mode=Mode.PREPARE)
    loco = Loco(r)
    assert loco.move(0.2, 0, 0) != 0
    r.mode = Mode.WALKING
    assert loco.move(0.2, 0, 0) == 0
    assert r.twist == (0.2, 0, 0)


def test_sim_records_every_call_and_integrates_pose():
    r = SimRobot(mode=Mode.WALKING)
    loco = Loco(r)
    loco.move(0.5, 0, 0)
    r.advance(2.0)
    assert abs(r.pose[0] - 1.0) < 1e-6
    loco.rotate_head(0.1, -0.2)
    assert r.head == (0.1, -0.2)
    assert [c[0] for c in r.calls] == [Api.MOVE, Api.ROTATE_HEAD]


def test_prepare_from_walking_zeroes_twist():
    r = SimRobot(mode=Mode.WALKING)
    loco = Loco(r)
    loco.move(0.3, 0, 0)
    assert loco.change_mode(Mode.PREPARE) == 0
    assert r.mode == Mode.PREPARE and r.twist == (0, 0, 0)


def test_sim_walking_only_reachable_from_prepare():
    r = SimRobot(mode=Mode.DAMPING)
    assert Loco(r).change_mode(Mode.WALKING) != 0
    r.mode = Mode.PREPARE
    assert Loco(r).change_mode(Mode.WALKING) == 0


def test_get_mode_reads_back():
    assert Loco(SimRobot(mode=Mode.WALKING)).get_mode() == Mode.WALKING


def test_gestures_need_standing_robot():
    r = SimRobot(mode=Mode.DAMPING)
    assert Loco(r).wave_hand(open_=True) != 0
    r.mode = Mode.WALKING
    assert Loco(r).wave_hand(open_=True) == 0 and Loco(r).handshake(open_=False) == 0
