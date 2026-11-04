from k1_chat.conversation import WakeGate


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_ignores_without_wake_word_then_window():
    c = Clock()
    g = WakeGate(window_s=20, clock=c)
    assert g.check("so the robot fell over") is None
    assert g.check("Hey, robot. Walk forward a bit.") == "Walk forward a bit"
    c.t = 10
    assert g.check("now turn left") == "now turn left"       # follow-up inside the window
    c.t = 29
    assert g.check("what time is it") == "what time is it"   # window was extended at t=10
    c.t = 50
    assert g.check("what time is it") is None


def test_default_needs_name_every_time():
    c = Clock()
    g = WakeGate(words="hey apollo,apollo", clock=c)
    assert g.check("Apollo, turn left a little") == "turn left a little"
    g.touch()  # robot spoke
    assert g.check("it fails") is None                      # room talk after the reply is ignored
    assert g.check("Hey Apollo.") == ""                     # bare name -> "Yes?"
    c.t = 5
    assert g.check("walk forward") == "walk forward"        # one command passes
    assert g.check("and the VLM is on the cloud") is None   # then the gate closes again
    g.check("apollo")
    c.t = 20
    assert g.check("walk forward") is None                  # armed only for arm_s


def test_wake_word_alone_and_mid_sentence():
    g = WakeGate(clock=Clock())
    assert g.check("Hey robot!") == ""
    assert g.check("ok so hey booster what do you see") == "ok so what do you see"


def test_empty_words_disable_nothing_passes():
    g = WakeGate(words="", clock=Clock())
    assert g.check("hey robot walk") is None


def test_clauses_splits_long_text():
    from k1_chat.robot_audio import clauses
    t = ("There's an obstacle ahead, a person straight ahead about a metre and a third away, "
         "another person to the left, a TV to the right, and a chair nearby.")
    parts = clauses(t)
    assert len(parts) > 1 and all(len(p) <= 120 for p in parts)
    assert " ".join(parts) == t
    assert clauses("Short one.") == ["Short one."]


def test_filler_around_name_is_bare_wake_and_arm():
    c = Clock()
    g = WakeGate(words="hey apollo,apollo", clock=c)
    assert g.check("He, Apollo") == ""
    assert g.check("20.") == "20."             # armed by the bare name
    assert g.check("and now?") is None
    g.arm()                                     # robot asked a question / keep_listening
    c.t = 5
    assert g.check("20 degrees") == "20 degrees"
    assert g.check("chatting with a friend") is None
