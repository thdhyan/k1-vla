"""Which free OpenRouter models take an image AND return a sensible tool call?

  uv run --with httpx --with python-dotenv python scripts/probe_vl_tools.py [image.jpg] [model ...]
No models given: every ':free' model whose OpenRouter metadata lists image input + tools.
Prints one line per model: latency, tool calls, text. Uses OPENROUTER_API_KEY from ./.env.
"""
import json, sys, time
from pathlib import Path

import httpx
from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from k1_chat.llm import image_part  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": p, "required": list(p)}}} for n, d, p in [
    ("turn", "Turn in place. Positive = left (counter-clockwise), negative = right.", {"angle_deg": {"type": "number"}}),
    ("walk", "Walk straight, metres, negative = backward.", {"distance_m": {"type": "number"}}),
    ("wave", "Wave the right hand.", {}),
]]


def candidates() -> list[str]:
    d = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    return sorted(m["id"] for m in d if m["id"].endswith(":free") and "tools" in (m.get("supported_parameters") or [])
                  and "image" in m.get("architecture", {}).get("input_modalities", []))


def main() -> None:
    args = sys.argv[1:]
    img = Path(args.pop(0)) if args and args[0].endswith((".jpg", ".jpeg")) else Path(__file__).parents[1] / "tests/fixtures/ball_right.jpg"
    key = dotenv_values(Path(__file__).parents[1] / ".env")["OPENROUTER_API_KEY"]
    msgs = [{"role": "system", "content": "You are a small humanoid robot. The image is your head camera. Act with tools."},
            {"role": "user", "content": [{"type": "text", "text": "Is there a ball? Which side? Turn to face it."},
                                         image_part(img.read_bytes())]}]
    for m in args or candidates():
        t = time.time()
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions", timeout=60,
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": m, "messages": msgs, "tools": TOOLS, "max_tokens": 2000})
            if r.status_code != 200:
                print(f"{m}: HTTP {r.status_code} {r.text[:120]}", flush=True)
                continue
            msg = r.json()["choices"][0]["message"]
            calls = [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")) for c in msg.get("tool_calls") or []]
            print(f"{m}: {time.time() - t:.1f}s calls={calls} text={(msg.get('content') or '')[:100]!r}", flush=True)
        except Exception as e:
            print(f"{m}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
