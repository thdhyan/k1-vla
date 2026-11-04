# scripts

Part of [k1-vla](https://github.com/thdhyan/k1-vla).

| File | What it does |
|---|---|
| `demo.sh` | One command for the talking-robot demo (A1 = "Apollo"). Run from the laptop, any directory. Runbook: docs/DEMO_RUNBOOK.md |
| `k1tools.sh` | Laptop side: robot tools + console. Morning order: check -> (robot on gantry/held) agent dry run -> agent live. |
| `k1voice.sh` | Laptop side: manage the on-robot local ASR/TTS. Stops the robot's audio service while ours runs |
| `k1voice_install.sh` | Install the local voice stack on K1s (fw 1.8.x): faster-whisper tiny.en (CPU, 3x faster than base.en) + Kokoro-82M TTS on the robot GPU |
| `probe_vl_tools.py` | Which free OpenRouter models take an image AND return a sensible tool call? |
| `serve_llm.sh` | Self-hosted OpenAI-compatible models for the K1 brain (vLLM in docker on a GPU box). Normally driven by the LLM panel |
