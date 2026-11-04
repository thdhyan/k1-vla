"""OpenRouter chat client (free models only) or a self-hosted OpenAI-compatible server (vLLM, llama.cpp).
tests/test_llm.py and tests/test_llm_tools.py define the contract."""
import base64
import os
import time

import httpx

from k1_chat import debug

OPENROUTER = "https://openrouter.ai/api/v1"


class LLMError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str = OPENROUTER,
                 retries: int = 3, backoff: float = 0.25, timeout: float = 60.0, sleep=time.sleep):
        # free-only rule applies to OpenRouter; a self-hosted server (our own GPU) costs nothing
        if "openrouter.ai" in base_url and not model.endswith(":free"):
            raise ValueError(f"only ':free' models allowed, got {model!r}")
        self.api_key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.retries, self.backoff, self.timeout, self._sleep = retries, backoff, timeout, sleep

    def chat(self, messages: list[dict], **kw) -> str:
        return self.complete(messages, **kw).get("content") or ""

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kw) -> dict:
        """Full assistant message (content and tool_calls). `tools` is OpenAI function-calling format."""
        t = time.time()
        try:
            msg = self._complete(messages, tools, **kw)
        except LLMError as e:
            debug.note(f"llm {self.model}", time.time() - t, ok=False, detail=str(e))
            raise
        debug.note(f"llm {self.model}", time.time() - t, detail=f"tool_calls={len(msg.get('tool_calls') or [])}")
        return msg

    def _complete(self, messages: list[dict], tools: list[dict] | None, **kw) -> dict:
        body = {"model": self.model, "messages": messages, **({"tools": tools} if tools else {}), **kw}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last = "no attempt"
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(self.backoff * 2 ** (attempt - 1))
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
            except httpx.TransportError as e:
                last = f"transport error: {e}"
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
            try:
                msg = r.json()["choices"][0]["message"]
                if not isinstance(msg, dict):
                    raise TypeError("message is not an object")
                return msg
            except (KeyError, IndexError, TypeError, ValueError) as e:
                raise LLMError(f"bad response shape: {e}") from e
        raise LLMError(f"gave up after {self.retries + 1} attempts: {last}")


def image_part(jpeg: bytes) -> dict:
    """User-content part carrying a JPEG for vision models."""
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def from_env(model_var: str = "K1_CHAT_MODEL") -> OpenRouterClient:
    """K1_CHAT_MODEL -> OpenRouter. With K1_CHAT_BASE_URL set (e.g. http://dl:8000/v1) -> that self-hosted server,
    key from K1_CHAT_API_KEY (vLLM accepts any). Same for K1_VISION_*."""
    model = os.environ.get(model_var)
    if not model:
        raise LLMError(f"{model_var} not set")
    prefix = model_var.removesuffix("_MODEL")
    base = os.environ.get(f"{prefix}_BASE_URL")
    if base:
        return OpenRouterClient(os.environ.get(f"{prefix}_API_KEY", "local"), model, base_url=base)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY not set")
    return OpenRouterClient(key, model)
