import httpx

from k1_chat.backend import Mode, SimRobot
from k1_chat.brain import Brain


class LLM:
    def __init__(self, msgs): self.msgs = list(msgs)
    def complete(self, messages, tools=None, **kw): return self.msgs.pop(0)


def call(name, args):
    return {"id": "1", "type": "function", "function": {"name": name, "arguments": args}}


def test_text_turn_runs_tool_on_sim():
    sim = SimRobot(mode=Mode.WALKING)
    b = Brain("A2", sim, LLM([{"content": None, "tool_calls": [call("set_head", '{"yaw_deg": 20, "pitch_deg": 0}')]},
                              {"content": "Looking left."}]), system="s")
    assert b.handle("look left") == "Looking left."
    assert sim.head[1] > 0
    kinds = [e["kind"] for e in b.log.since(0)]
    assert kinds[0] == "heard" and kinds[-1] == "said"


def test_spoken_stop_estops_without_llm():
    sim = SimRobot(mode=Mode.WALKING)
    b = Brain("A2", sim, LLM([]), system="s")
    assert "stop" in b.handle("STOP!").lower()
    assert sim.mode == Mode.PREPARE and b.arbiter.estopped


def test_console_serves_and_estops():
    sim = SimRobot(mode=Mode.WALKING)
    b = Brain("A2", sim, LLM([]), system="s")
    url = b.serve_console("127.0.0.1", 0)
    try:
        assert "A2" in httpx.get(url).text
        httpx.post(url + "api/estop")
        assert sim.mode == Mode.PREPARE
        st = httpx.get(url + "api/events").json()["state"]
        assert st["mode"] == "PREPARE" and st["estopped"]
    finally:
        b.close()


def test_brain_chains_several_dependent_tool_calls():
    # search: look, head left, look, turn left, wave -- each step its own round
    sim = SimRobot(mode=Mode.WALKING)
    steps = [call("set_head", '{"yaw_deg": 40, "pitch_deg": 0}'), call("set_head", '{"yaw_deg": -40, "pitch_deg": 0}'),
             call("set_head", '{"yaw_deg": 0, "pitch_deg": 0}'), call("turn", '{"angle_deg": 30}'),
             call("set_head", '{"yaw_deg": 10, "pitch_deg": 0}'), call("wave", '{"seconds": 1}')]
    b = Brain("A2", sim, LLM([{"content": None, "tool_calls": [c]} for c in steps] + [{"content": "Hi there!"}]), system="s")
    assert b.handle("find the person and wave") == "Hi there!"
    ran = [e["name"] for e in b.log.since(0) if e["kind"] == "tool_result"]
    assert ran == ["set_head", "set_head", "set_head", "turn", "set_head", "wave"]


def test_brain_turns_run_one_at_a_time():
    import threading, time
    sim = SimRobot(mode=Mode.WALKING)
    active, overlap = [], []

    class Slow:
        def complete(self, messages, tools=None, **kw):
            active.append(1)
            overlap.append(len(active) > 1)
            time.sleep(0.05)
            active.pop()
            return {"content": "ok"}
    b = Brain("A2", sim, Slow(), system="s")
    ts = [threading.Thread(target=b.handle, args=(t,)) for t in ("wave", "look for a person")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert overlap == [False, False]
