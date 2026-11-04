# tests

Part of [k1-vla](https://github.com/thdhyan/k1-vla).

| File | What it does |
|---|---|
| `test_arbiter.py` |  |
| `test_asr_adapter.py` | NemotronAsr adapter -> nemo-asrd daemon (K1_CHAT_ASR=nemo). Tests first: plan WAKEWORD_NEMO.md §3.3 / booster_ws PLAN_VOICE_VLM §9.6. |
| `test_asr_server.py` |  |
| `test_audio.py` |  |
| `test_backend.py` |  |
| `test_booster_robot.py` |  |
| `test_brain.py` |  |
| `test_console.py` |  |
| `test_conversation.py` |  |
| `test_conversation_tools.py` |  |
| `test_debug.py` |  |
| `test_events_safety.py` |  |
| `test_fleet.py` |  |
| `test_kokoro_mem_limit.py` | KokoroTts must cap the onnxruntime CUDA arena (unified-memory RAM on the Orin) — the default arena starves nemo-asrd and the daemon gets OOM |
| `test_llm.py` |  |
| `test_llm_panel.py` |  |
| `test_llm_tools.py` |  |
| `test_modes.py` |  |
| `test_robot_tools.py` |  |
| `test_script_tools.py` |  |
| `test_tools.py` |  |
| `test_vision.py` |  |
| `test_wake.py` |  |
