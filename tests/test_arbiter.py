from k1_chat.arbiter import Twist, VelocityArbiter


class Clock:
    t = 0.0
    def __call__(self): return self.t


def make():
    c = Clock()
    return VelocityArbiter(max_vx=0.4, max_vy=0.2, max_vyaw=0.5, chat_timeout=1.0, manual_hold=2.0, clock=c), c


def test_idle_is_zero():
    a, _ = make(); assert a.output() == Twist()

def test_chat_command_passes_and_is_clamped():
    a, _ = make(); a.chat(Twist(vx=5, vy=-5, vyaw=5))
    assert a.output() == Twist(0.4, -0.2, 0.5)

def test_chat_command_times_out_to_zero():
    a, c = make(); a.chat(Twist(vx=0.3)); c.t = 1.5
    assert a.output() == Twist()

def test_manual_overrides_chat():
    a, _ = make(); a.chat(Twist(vx=0.3)); a.manual(Twist(vx=-0.1))
    assert a.output() == Twist(vx=-0.1)

def test_chat_blocked_during_manual_hold_then_resumes():
    a, c = make(); a.manual(Twist(vx=0.1)); c.t = 0.5; a.chat(Twist(vx=0.3))
    assert a.output().vx != 0.3
    c.t = 2.6; a.chat(Twist(vx=0.3))
    assert a.output() == Twist(vx=0.3)

def test_manual_zero_stop_still_blocks_chat():
    a, c = make(); a.manual(Twist()); a.chat(Twist(vx=0.3))
    assert a.output() == Twist()

def test_estop_latches_until_release():
    a, _ = make(); a.estop(); a.chat(Twist(vx=0.3)); assert a.output() == Twist()
    a.release(); a.chat(Twist(vx=0.3)); assert a.output() == Twist(vx=0.3)

def test_manual_is_not_clamped_by_chat_limits():
    a, _ = make(); a.manual(Twist(vx=0.8)); assert a.output().vx == 0.8
