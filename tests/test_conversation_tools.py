from k1_chat.conversation import Conversation, is_stop
from k1_chat.events import EventLog
from k1_chat.tools import Registry, Tool


def call(name, args="{}", id_="c1"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}


class ScriptedLLM:
    """complete() returns the scripted assistant messages in order."""
    def __init__(self, msgs):
        self.msgs, self.calls = list(msgs), []

    def complete(self, messages, tools=None, **kw):
        self.calls.append(([dict(m) for m in messages], tools))
        return self.msgs.pop(0)


def reg_with(log=None):
    ran = []
    r = Registry(log)
    r.add(Tool("wave", "Wave.", {}, lambda: ran.append("wave") or "wave done"))
    return r, ran


def test_tool_call_runs_and_result_goes_back_to_model():
    reg, ran = reg_with()
    llm = ScriptedLLM([{"role": "assistant", "content": None, "tool_calls": [call("wave")]},
                       {"role": "assistant", "content": "Hello there!"}])
    c = Conversation(llm, system="s", tools=reg)
    assert c.reply("wave at me") == "Hello there!"
    assert ran == ["wave"]
    second = llm.calls[1][0]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "wave"
    assert second[-1] == {"role": "tool", "tool_call_id": "c1", "content": "wave done"}
    assert llm.calls[0][1] == reg.schemas()


def test_history_keeps_only_text_turns():
    reg, _ = reg_with()
    llm = ScriptedLLM([{"content": None, "tool_calls": [call("wave")]}, {"content": "Done."}])
    c = Conversation(llm, system="s", tools=reg)
    c.reply("wave")
    assert c.history == [{"role": "user", "content": "wave"}, {"role": "assistant", "content": "Done."}]


def test_tool_calls_per_turn_are_capped():
    reg, ran = reg_with()
    loop = {"content": None, "tool_calls": [call("wave", id_=str(i)) for i in range(2)]}
    llm = ScriptedLLM([loop, loop, loop, {"content": "ok"}])
    c = Conversation(llm, system="s", tools=reg, max_tool_calls=3)
    c.reply("wave a lot")
    assert len(ran) == 3


def test_model_that_never_answers_gets_fallback_after_rounds():
    reg, _ = reg_with()
    llm = ScriptedLLM([{"content": None, "tool_calls": [call("wave", id_=str(i))]} for i in range(10)])
    c = Conversation(llm, system="s", tools=reg, max_tool_calls=20, max_rounds=3)
    assert c.reply("x")                                  # spoken fallback, never None
    assert len(llm.calls) == 3


def test_events_logged_for_console():
    log = EventLog()
    reg, _ = reg_with(log)
    llm = ScriptedLLM([{"content": None, "tool_calls": [call("wave")]}, {"content": "Hi."}])
    Conversation(llm, system="s", tools=reg, log=log).reply("hey")
    kinds = [e["kind"] for e in log.since(0)]
    assert kinds == ["llm_request", "llm_reply", "tool_call", "tool_result", "llm_request", "llm_reply"]


def test_without_tools_old_chat_path_is_used():
    class Old:
        def chat(self, messages, **kw): return "plain"
    assert Conversation(Old(), system="s").reply("x") == "plain"


def test_stop_phrases():
    for t in ("Stop!", "stop walking", "please STOP now", "halt", "freeze"):
        assert is_stop(t), t
    assert not is_stop("hello there") and not is_stop("unstoppable")


def test_one_call_per_round_runs_only_the_first():
    reg, ran = reg_with()
    reg.add(Tool("look", "Look.", {}, lambda: ran.append("look") or "a person on the left"))
    llm = ScriptedLLM([{"role": "assistant", "content": None, "tool_calls": [call("look", id_="a"), call("wave", id_="b")]},
                       {"role": "assistant", "content": "Found you."}])
    conv = Conversation(llm, system="s", tools=reg, one_call_per_round=True)
    assert conv.reply("find me and wave") == "Found you."
    assert ran == ["look"]
    tool_msgs = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"] and "not run" in tool_msgs[1]["content"]


def test_model_thinking_is_logged_for_console():
    log = EventLog()
    reg, _ = reg_with(log)
    llm = ScriptedLLM([{"role": "assistant", "content": "Hi.", "reasoning": "The person greets me; no tool needed."}])
    Conversation(llm, system="s", tools=reg, log=log).reply("hello")
    assert [e["thinking"] for e in log.since(0) if e["kind"] == "llm_reply"] == ["The person greets me; no tool needed."]
