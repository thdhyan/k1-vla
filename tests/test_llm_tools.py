import json

import httpx, respx
from k1_chat.llm import OpenRouterClient, image_part

URL = "https://openrouter.ai/api/v1/chat/completions"


@respx.mock
def test_complete_returns_full_message_with_tool_calls_and_sends_tools():
    call = {"id": "c1", "type": "function", "function": {"name": "turn", "arguments": '{"angle_deg": 30}'}}
    route = respx.post(URL).mock(return_value=httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}))
    tools = [{"type": "function", "function": {"name": "turn", "parameters": {"type": "object"}}}]
    msg = OpenRouterClient("k", "m:free").complete([{"role": "user", "content": "x"}], tools=tools)
    assert msg["tool_calls"][0]["function"]["name"] == "turn"
    assert json.loads(route.calls[0].request.content)["tools"] == tools


@respx.mock
def test_complete_without_tools_sends_no_tools_key():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))
    assert OpenRouterClient("k", "m:free").complete([{"role": "user", "content": "x"}])["content"] == "hi"
    assert "tools" not in json.loads(route.calls[0].request.content)


def test_image_part_is_openai_style_data_url():
    p = image_part(b"\xff\xd8jpeg")
    assert p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg;base64,")
