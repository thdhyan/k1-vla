"""LLM tool registry: narrow flat tools -> OpenAI function schemas; run() never raises, errors go back to the model
as text. Numbers are clamped to the schema's minimum/maximum. tests/test_tools.py."""
import json
from dataclasses import dataclass
from typing import Callable

from k1_chat.events import EventLog


@dataclass
class Tool:
    name: str
    description: str
    params: dict  # JSON-schema properties; every key is required
    fn: Callable[..., str]


_TYPES = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,)}


class Registry:
    def __init__(self, log: EventLog | None = None):
        self._tools: dict[str, Tool] = {}
        self._log = log
        self.enabled: Callable[[str], bool] = lambda name: True

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return [n for n in self._tools if self.enabled(n)]

    def params(self, name: str) -> dict:
        return self._tools[name].params

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {
                    "name": t.name, "description": t.description,
                    "parameters": {"type": "object", "properties": t.params, "required": list(t.params)}}}
                for t in self._tools.values() if self.enabled(t.name)]

    def _args(self, tool: Tool, raw: str) -> dict:
        args = json.loads(raw) if raw and raw.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        out = {}
        for k, spec in tool.params.items():
            if k not in args:
                raise ValueError(f"missing argument {k!r}")
            v = args[k]
            ok = _TYPES.get(spec.get("type"), (object,))
            if not isinstance(v, ok) or (isinstance(v, bool) and bool not in ok):
                raise ValueError(f"{k!r} must be {spec.get('type')}")
            if "enum" in spec and v not in spec["enum"]:
                raise ValueError(f"{k!r} must be one of {spec['enum']}")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = min(spec.get("maximum", v), max(spec.get("minimum", v), v))
            out[k] = v
        return out

    def run(self, name: str, raw_args: str) -> str:
        tool = self._tools.get(name)
        if tool is None or not self.enabled(name):
            result = f"error: no tool {name!r} available"
            args = None
        else:
            try:
                args = self._args(tool, raw_args)
            except (ValueError, TypeError) as e:  # json.JSONDecodeError is a ValueError
                args, result = None, f"error: bad arguments for {name}: {e}"
            else:
                if self._log:
                    self._log.add("tool_call", name=name, args=args)
                try:
                    result = str(tool.fn(**args))
                except Exception as e:
                    result = f"error: {name} failed: {type(e).__name__}: {e}"
        if self._log:
            if args is None:
                self._log.add("tool_call", name=name, raw=raw_args)
            self._log.add("tool_result", name=name, result=result)
        return result
