"""`macro` and `run_script`: let the model chain robot tools in one call. tests/test_script_tools.py.

Both only call tools already in the registry (Registry.run), so every guard stays: arbiter clamps, ModeFSM, e-stop
latch, gesture busy checks. Neither can reach the OS.

run_script is a small Python subset, checked on the AST before it runs: statements if/for/while/break/continue/pass and
assignment; expressions are names, numbers, strings, lists, comparisons, and/or/not, + - / //, f-strings without format
specs, and calls to the robot tools or the helpers below. No imports, attributes, subscripts, def/lambda/class, `*`,
`**`, `%`, or names starting with "_". At run time: MAX_LINES executed lines, MAX_CALLS tool calls, MAX_S seconds, no
string or list over MAX_LEN, and the script ends as soon as the e-stop latches.
"""
import ast
import json
import sys
import time

from k1_chat.tools import Registry, Tool

MAX_LINES, MAX_CALLS, MAX_S, MAX_LEN, MAX_STEPS = 2000, 30, 90.0, 10_000, 10
NESTED = ("macro", "run_script")
_OK = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Call,
       ast.keyword, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass, ast.Compare, ast.BoolOp,
       ast.UnaryOp, ast.BinOp, ast.IfExp, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd, ast.Add, ast.Sub, ast.Div,
       ast.FloorDiv, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.List, ast.Tuple,
       ast.JoinedStr, ast.FormattedValue)
_STOPPED = "error: e-stop is active, script ended"


class ScriptError(Exception):
    pass


def _check(code: str, callable_names: set[str]) -> ast.Module:
    tree = ast.parse(code, "<script>", "exec")
    for node in ast.walk(tree):
        if not isinstance(node, _OK):
            raise ScriptError(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ScriptError(f"line {node.lineno}: names starting with _ are not allowed")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in callable_names):
            raise ScriptError(f"line {node.lineno}: you can only call {sorted(callable_names)}")
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            raise ScriptError(f"line {node.lineno}: f-string format specs are not allowed")
    return tree


def add_script_tools(reg: Registry, stopped=lambda: False, sleep=time.sleep, clock=time.monotonic) -> None:
    def macro(steps: list) -> str:
        lines = []
        for i, step in enumerate(steps[:MAX_STEPS], 1):
            name, args = (step.get("tool"), step.get("args") or {}) if isinstance(step, dict) else (None, {})
            if name in NESTED:
                return "\n".join(lines + [f"{i}. {name}: not allowed inside a macro"])
            if stopped():
                return "\n".join(lines + [_STOPPED])
            result = reg.run(str(name), json.dumps(args))
            lines.append(f"{i}. {name}: {result}")
            if result.startswith(("error", "refused", "stopped")):
                return "\n".join(lines + [f"stopped at step {i}"])
        return "\n".join(lines) or "error: no steps"

    def run_script(code: str) -> str:
        tools = [n for n in reg.names() if n not in NESTED]
        out, calls, t0 = [], [0], clock()

        def tool_fn(name):
            keys = list(reg.params(name))

            def call(*a, **kw):
                if stopped():
                    raise ScriptError(_STOPPED[7:])
                if calls[0] >= MAX_CALLS:
                    raise ScriptError(f"tool call limit ({MAX_CALLS}) reached")
                if clock() - t0 > MAX_S:
                    raise ScriptError(f"time limit ({MAX_S:.0f} s) reached")
                calls[0] += 1
                return reg.run(name, json.dumps({**dict(zip(keys, a)), **kw}))
            return call

        def mul(a, b):  # `*` is not allowed in scripts: 's' * n would build huge strings
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)):
                raise ScriptError("mul() takes numbers")
            return a * b

        helpers = {
            "print": lambda *a: out.append(" ".join(map(str, a))),
            "contains": lambda text, word: str(word).lower() in str(text).lower(),
            "lower": lambda s: str(s).lower(),
            "sleep": lambda s: sleep(min(max(float(s), 0.0), 5.0)),
            "mul": mul,
            "len": len, "range": lambda *a: range(*a), "min": min, "max": max, "abs": abs, "round": round,
            "int": int, "float": float, "str": str,
        }
        env = {"__builtins__": {}, **helpers, **{n: tool_fn(n) for n in tools}}
        try:
            prog = compile(_check(code, set(env) - {"__builtins__"}), "<script>", "exec")
        except (SyntaxError, ScriptError) as e:
            return f"error: {e}"

        lines = [0]

        def local(frame, event, arg):
            if event == "line":
                lines[0] += 1
                if lines[0] > MAX_LINES:
                    raise ScriptError(f"too many steps (over {MAX_LINES} lines run); loops need a tool call or break")
                for v in frame.f_locals.values():
                    if isinstance(v, (str, list, tuple)) and len(v) > MAX_LEN:
                        raise ScriptError(f"value too big (over {MAX_LEN} items)")
            return local

        def trace(frame, event, arg):
            return local if frame.f_code.co_filename == "<script>" else None

        old = sys.gettrace()
        sys.settrace(trace)
        try:
            exec(prog, env)
            status = f"done ({calls[0]} tool calls)"
        except ScriptError as e:
            status = f"error: {e}"
        except Exception as e:
            status = f"error: {type(e).__name__}: {e}"
        finally:
            sys.settrace(old)
        text = "\n".join(out[-40:] + [status])
        return text[-1500:]

    reg.add(Tool("macro", "Run several robot tool calls in order as one step; stops at the first error or refusal. "
                 "Use for a fixed sequence (e.g. turn then wave).",
                 {"steps": {"type": "array", "maxItems": MAX_STEPS, "description": "list of {tool, args}",
                            "items": {"type": "object", "properties": {"tool": {"type": "string"},
                                                                      "args": {"type": "object"}}}}}, macro))
    reg.add(Tool("run_script", "Run a short Python script that calls robot tools as functions with keyword arguments, "
                 "e.g. look(question='...'), turn(angle_deg=90), wave(seconds=2); each returns the tool's text. Use for "
                 "loops like searching until something is seen. Allowed: if/for/while/break, range, len, min, max, abs, "
                 "round, int, float, str, contains(text, word), lower(s), mul(a, b), sleep(s), print(...). Not allowed: "
                 "import, attributes (x.y), indexing (x[0]), def, lambda, *, %. Limits: 30 tool calls, 90 s. "
                 "print() what you found; that text comes back to you.",
                 {"code": {"type": "string", "description": "the Python script"}}, run_script))
