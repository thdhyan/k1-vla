"""KokoroTts must cap the onnxruntime CUDA arena (unified-memory RAM on the Orin) — the
default arena starves nemo-asrd and the daemon gets OOM-killed mid-transcribe.
See docs/PLAN_VOICE_VLM_2026-10-28.md §9.7."""
import sys
import types

import pytest


@pytest.fixture()
def fake_ort(monkeypatch):
    captured = {}

    class FakeSess:
        def __init__(self, path, providers=None):
            captured["providers"] = providers

        def get_providers(self):
            return ["CUDAExecutionProvider"]

    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = FakeSess
    kokoro = types.ModuleType("kokoro_onnx")

    class FakeKokoro:
        @staticmethod
        def from_session(sess, voices):
            return object()

    kokoro.Kokoro = FakeKokoro
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", kokoro)
    return captured


def test_cuda_provider_is_capped_by_default(fake_ort, monkeypatch):
    monkeypatch.delenv("K1_KOKORO_CUDA_MEM_LIMIT", raising=False)
    from k1_chat.robot_audio import KokoroTts

    KokoroTts()
    providers = fake_ort["providers"]
    assert providers[0][0] == "CUDAExecutionProvider", providers
    opts = providers[0][1]
    assert opts["gpu_mem_limit"] > 0
    assert opts["arena_extend_strategy"] == "kSameAsRequested"
    assert providers[1] == "CPUExecutionProvider"  # fallback kept


def test_mem_limit_env_overrides(fake_ort, monkeypatch):
    monkeypatch.setenv("K1_KOKORO_CUDA_MEM_LIMIT", str(128 * 1024 * 1024))
    from k1_chat.robot_audio import KokoroTts

    KokoroTts()
    assert fake_ort["providers"][0][1]["gpu_mem_limit"] == 128 * 1024 * 1024


def test_cpu_device_uses_cpu_provider_only(fake_ort, monkeypatch):
    monkeypatch.setenv("K1_KOKORO_DEVICE", "cpu")
    from k1_chat.robot_audio import KokoroTts

    KokoroTts()
    assert fake_ort["providers"] == ["CPUExecutionProvider"]
