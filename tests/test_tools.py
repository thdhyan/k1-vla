from k1_chat.events import EventLog
from k1_chat.tools import Registry, Tool


def make(log=None):
    seen = []
    reg = Registry(log)
    reg.add(Tool("turn", "Rotate in place.", {"angle_deg": {"type": "number", "minimum": -180, "maximum": 180}},
                 lambda angle_deg: seen.append(angle_deg) or f"turned {angle_deg}"))
    reg.add(Tool("wave", "Wave.", {}, lambda: "waved"))
    return reg, seen


def test_schemas_are_openai_function_format():
    reg, _ = make()
    s = {t["function"]["name"]: t for t in reg.schemas()}
    assert s["turn"]["type"] == "function"
    assert s["turn"]["function"]["parameters"]["required"] == ["angle_deg"]
    assert s["wave"]["function"]["parameters"] == {"type": "object", "properties": {}, "required": []}


def test_run_parses_json_and_clamps_to_schema_limits():
    reg, seen = make()
    assert reg.run("turn", '{"angle_deg": 400}').startswith("turned 180")
    assert seen == [180]


def test_errors_come_back_as_text_not_exceptions():
    reg, _ = make()
    assert reg.run("fly", "{}").startswith("error")
    assert reg.run("turn", "not json").startswith("error")
    assert reg.run("turn", "{}").startswith("error")              # missing required
    assert reg.run("turn", '{"angle_deg": "left"}').startswith("error")
    reg.add(Tool("boom", "x", {}, lambda: 1 / 0))
    assert "ZeroDivisionError" in reg.run("boom", "{}")


def test_empty_arguments_string_means_no_args():
    reg, _ = make()
    assert reg.run("wave", "") == "waved"


def test_disabled_tool_is_refused_and_hidden():
    reg, seen = make()
    reg.enabled = lambda name: name != "turn"
    assert reg.run("turn", '{"angle_deg": 10}').startswith("error") and seen == []
    assert [t["function"]["name"] for t in reg.schemas()] == ["wave"]


def test_calls_and_results_are_logged():
    log = EventLog()
    reg, _ = make(log)
    reg.run("turn", '{"angle_deg": 10}')
    kinds = [e["kind"] for e in log.since(0)]
    assert kinds == ["tool_call", "tool_result"]
    assert log.since(0)[0]["args"] == {"angle_deg": 10}
