import json

from k1_chat.events import EventLog
from k1_chat.script_tools import add_script_tools
from k1_chat.tools import Registry, Tool


def make(stopped=lambda: False, looks=("nobody", "nobody", "a person on the left")):
    ran, seen = [], list(looks)
    reg = Registry(EventLog())
    reg.add(Tool("look", "Look.", {"question": {"type": "string"}}, lambda question: ran.append("look") or seen.pop(0)))
    reg.add(Tool("turn", "Turn.", {"angle_deg": {"type": "number", "minimum": -180, "maximum": 180}},
                 lambda angle_deg: ran.append(("turn", angle_deg)) or f"turned about {angle_deg:.0f} degrees"))
    reg.add(Tool("wave", "Wave.", {"seconds": {"type": "number"}}, lambda seconds: ran.append("wave") or "wave done"))
    reg.add(Tool("fail", "Fails.", {}, lambda: "refused: gestures need walking mode"))
    add_script_tools(reg, stopped=stopped, sleep=lambda s: None)
    return reg, ran


def script(reg, code):
    return reg.run("run_script", json.dumps({"code": code}))


def test_macro_runs_steps_in_order():
    reg, ran = make()
    out = reg.run("macro", json.dumps({"steps": [{"tool": "turn", "args": {"angle_deg": 90}},
                                                 {"tool": "wave", "args": {"seconds": 2}}]}))
    assert ran == [("turn", 90), "wave"] and "1. turn: turned about 90 degrees" in out and "2. wave: wave done" in out


def test_macro_stops_at_first_refusal():
    reg, ran = make()
    out = reg.run("macro", json.dumps({"steps": [{"tool": "fail", "args": {}}, {"tool": "wave", "args": {"seconds": 1}}]}))
    assert ran == [] and "stopped at step 1" in out


def test_macro_cannot_nest():
    reg, _ = make()
    out = reg.run("macro", json.dumps({"steps": [{"tool": "run_script", "args": {"code": "wave(seconds=1)"}}]}))
    assert "not allowed" in out


def test_script_search_loop_then_wave():
    reg, ran = make()
    out = script(reg, """
for i in range(4):
    seen = look(question="is there a person?")
    if contains(seen, "person"):
        print("found after", i, "turns")
        wave(seconds=2)
        break
    turn(angle_deg=90)
""")
    assert ran == ["look", ("turn", 90), "look", ("turn", 90), "look", "wave"]
    assert "found after 2 turns" in out


def test_script_positional_args_and_clamping():
    reg, ran = make()
    script(reg, "turn(500)")
    assert ran == [("turn", 180)]


def test_script_rejects_escape_hatches():
    reg, ran = make()
    for code in ("import os", "().__class__", "x = look.__globals__", "open('/etc/passwd')", "__import__('os')",
                 "eval('1')", "def f():\n    pass", "lambda: 1", "x = [1][0]", "f'{1:99999999}'", "x = 'a' * 9",
                 "with x:\n    pass", "run_script(code='1')", "macro(steps=[])"):
        assert script(reg, code).startswith("error"), code
    assert ran == []


def test_script_line_budget_stops_busy_loop():
    reg, _ = make()
    assert "too many steps" in script(reg, "while True:\n    pass")


def test_script_stops_runaway_string_growth():
    reg, _ = make()
    assert "too big" in script(reg, "s = 'aaaaaaaaaa'\nwhile True:\n    s = s + s")


def test_script_tool_call_cap():
    reg, ran = make(looks=["nobody"] * 100)
    assert "tool call limit" in script(reg, "while True:\n    look(question='x')")
    assert len(ran) == 30


def test_script_aborts_on_estop():
    stop = []
    reg, ran = make(stopped=lambda: bool(stop))
    reg.add(Tool("press", "x", {}, lambda: stop.append(1) or "ok"))
    out = script(reg, "press()\nturn(angle_deg=10)")
    assert "e-stop" in out and ran == []


def test_script_runtime_error_is_reported():
    reg, _ = make()
    assert "error" in script(reg, "x = 1 / 0")
