from k1_chat.conversation import Conversation, run_voice_chat


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, **kw):
        self.calls.append((list(messages), kw))
        return self.replies.pop(0)


def test_reply_keeps_history_and_system_prompt():
    llm = FakeLLM(["Hi there.", "Fine."])
    c = Conversation(llm, system="be brief")
    assert c.reply("hello") == "Hi there."
    assert c.reply("how are you") == "Fine."
    msgs = llm.calls[1][0]
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert [m["content"] for m in msgs[1:]] == ["hello", "Hi there.", "how are you"]


def test_history_is_trimmed_to_max_turns():
    llm = FakeLLM(["a", "b", "c"])
    c = Conversation(llm, system="s", max_turns=2)
    for t in ("1", "2", "3"):
        c.reply(t)
    msgs = llm.calls[2][0]
    assert [m["content"] for m in msgs[1:]] == ["2", "b", "3"]


def test_llm_error_gives_spoken_fallback_and_drops_the_turn():
    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("down")
    c = Conversation(Boom(), system="s")
    assert "sorry" in c.reply("hello").lower()
    assert c.history == []


def test_run_voice_chat_speaks_each_reply():
    spoken = []
    run_voice_chat(iter(["hello", "bye"]), Conversation(FakeLLM(["Hi.", "Goodbye."]), system="s"), spoken.append)
    assert spoken == ["Hi.", "Goodbye."]


def test_default_system_prompt_comes_from_file():
    from k1_chat.conversation import SYSTEM
    assert "robot" in SYSTEM.lower() and len(SYSTEM) > 20


def test_load_system_appends_every_knowledge_file_in_name_order(tmp_path):
    from k1_chat.conversation import load_system
    (tmp_path / "system_prompt.md").write_text("Persona.\n")
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "b.md").write_text("About B.\n")
    (tmp_path / "knowledge" / "a.md").write_text("About A.\n")
    (tmp_path / "knowledge" / "notes.txt").write_text("ignored")
    out = load_system(tmp_path)
    assert out.startswith("Persona.")
    assert out.index("About A.") < out.index("About B.") and "ignored" not in out


def test_load_system_without_knowledge_dir_is_just_the_persona(tmp_path):
    from k1_chat.conversation import load_system
    (tmp_path / "system_prompt.md").write_text("Persona.\n")
    assert load_system(tmp_path) == "Persona."


def test_default_system_prompt_knows_booster_and_minnesota():
    from k1_chat.conversation import SYSTEM
    assert "Booster Robotics" in SYSTEM and "University of Minnesota" in SYSTEM


def test_default_memory_is_twenty_turns():
    assert Conversation(FakeLLM([]), system="s")._max == 20


def test_history_survives_restart_via_file(tmp_path):
    f = tmp_path / "history.json"
    c = Conversation(FakeLLM(["Hi there."]), system="s", history_file=f)
    c.reply("hello")
    llm = FakeLLM(["Fine."])
    c2 = Conversation(llm, system="s", history_file=f)
    c2.reply("how are you")
    assert [m["content"] for m in llm.calls[0][0][1:]] == ["hello", "Hi there.", "how are you"]


def test_failed_turn_is_not_saved(tmp_path):
    f = tmp_path / "history.json"

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("down")
    Conversation(Boom(), system="s", history_file=f).reply("hello")
    assert Conversation(FakeLLM([]), system="s", history_file=f).history == []


def test_corrupt_history_file_is_ignored(tmp_path):
    f = tmp_path / "history.json"
    f.write_text("{not json")
    assert Conversation(FakeLLM([]), system="s", history_file=f).history == []


def test_llm_call_leaves_room_for_reasoning_tokens():
    llm = FakeLLM(["Hi."])
    Conversation(llm, system="s").reply("hello")
    kw = llm.calls[0][1]
    assert kw["max_tokens"] >= 1500 and "reasoning" not in kw


import threading
import time

from k1_chat.conversation import run_async_voice_chat


class SlowLLM:
    """First call blocks until released; later calls answer at once."""
    def __init__(self):
        self.release, self.calls = threading.Event(), []

    def chat(self, messages, **kw):
        self.calls.append([m["content"] for m in messages[1:]])
        if len(self.calls) == 1:
            self.release.wait(2)
        return f"answer {len(self.calls)}"


def test_new_utterance_supersedes_pending_llm_call():
    llm, spoken = SlowLLM(), []
    conv = Conversation(llm, system="s")

    def texts():
        yield "first"
        while not llm.calls:          # first call is now in flight
            time.sleep(0.01)
        yield "second"
        time.sleep(0.2)
        llm.release.set()             # stale first reply arrives late

    run_async_voice_chat(texts(), conv, spoken.append)
    assert spoken == ["answer 2"]                     # stale reply never spoken
    assert llm.calls[1] == ["first second"]           # user turns merged for the new call
    assert [m["role"] for m in conv.history] == ["user", "assistant"]


def test_async_chat_answers_each_utterance_when_not_overlapping():
    spoken = []

    def texts():
        yield "a"
        while len(spoken) < 1:
            time.sleep(0.01)
        yield "b"

    run_async_voice_chat(texts(), Conversation(FakeLLM(["x", "y"]), system="s"), spoken.append)
    assert spoken == ["x", "y"]
