import httpx, pytest, respx
from k1_chat.llm import LLMError, OpenRouterClient, from_env

URL = "https://openrouter.ai/api/v1/chat/completions"


@respx.mock
def test_chat_returns_text_and_sends_auth_and_model():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))
    c = OpenRouterClient("sk-test", "x/y:free")
    assert c.chat([{"role": "user", "content": "hello"}]) == "hi"
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer sk-test"
    assert b'"model":"x/y:free"' in req.content.replace(b" ", b"")


@respx.mock
def test_chat_retries_on_429_then_succeeds():
    respx.post(URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})])
    assert OpenRouterClient("k", "m:free").chat([{"role": "user", "content": "x"}]) == "ok"


@respx.mock
def test_chat_raises_llmerror_after_retries():
    respx.post(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(LLMError):
        OpenRouterClient("k", "m:free").chat([{"role": "user", "content": "x"}])


def test_refuses_paid_model():
    with pytest.raises(ValueError):
        OpenRouterClient("k", "openrouter/auto")


def test_from_env_reads_key_and_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    monkeypatch.setenv("K1_CHAT_MODEL", "a/b:free")
    assert from_env() is not None


@pytest.mark.net
def test_live_free_model_answers():
    from dotenv import load_dotenv; load_dotenv()
    assert from_env().chat([{"role": "user", "content": "Reply with the word pong."}])


@respx.mock
def test_self_hosted_server_allows_any_model_name():
    route = respx.post("http://dl:8000/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}))
    assert OpenRouterClient("local", "nemotron-bolt", base_url="http://dl:8000/v1").chat([{"role": "user", "content": "x"}]) == "hi"
    assert route.called


def test_refuses_paid_model_even_with_explicit_openrouter_url():
    with pytest.raises(ValueError):
        OpenRouterClient("k", "openrouter/auto", base_url="https://openrouter.ai/api/v1/")


def test_from_env_uses_base_url_without_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("K1_CHAT_MODEL", "nemotron-bolt")
    monkeypatch.setenv("K1_CHAT_BASE_URL", "http://dl:8000/v1")
    c = from_env()
    assert c.base_url == "http://dl:8000/v1" and c.model == "nemotron-bolt"


def test_from_env_vision_has_its_own_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk")
    monkeypatch.setenv("K1_CHAT_BASE_URL", "http://dl:8000/v1")
    monkeypatch.setenv("K1_VISION_MODEL", "g/v:free")
    monkeypatch.delenv("K1_VISION_BASE_URL", raising=False)
    assert "openrouter.ai" in from_env("K1_VISION_MODEL").base_url


@pytest.mark.onprem
def test_live_self_hosted_model_calls_turn_tool():
    """Server up first (scripts/serve_llm.sh). K1_CHAT_BASE_URL=http://<host>:8000/v1 K1_CHAT_MODEL=nemotron-bolt pytest -m onprem"""
    import json, time
    from dotenv import load_dotenv; load_dotenv()
    tools = [{"type": "function", "function": {"name": "turn", "description": "Turn the robot in place. Positive = left.",
              "parameters": {"type": "object", "properties": {"angle_deg": {"type": "number"}}, "required": ["angle_deg"]}}}]
    t = time.time()
    msg = from_env().complete([{"role": "system", "content": "You are a small humanoid robot. Act with tools."},
                               {"role": "user", "content": "Turn left by 90 degrees."}], tools=tools, max_tokens=1000)
    print(f"latency {time.time() - t:.2f}s")
    call = msg["tool_calls"][0]["function"]
    assert call["name"] == "turn" and json.loads(call["arguments"])["angle_deg"] > 0
