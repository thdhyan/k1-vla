# K1 Chatbot + Policy Loading — Plan

Branch `feat/k1-chatbot` (orphan, independent history) in worktree `~/Projects/wt-k1-chat`.
No files shared with `booster_ws`. Reads `booster_ws/sdk/booster_robotics_sdk` and talks to robots only over the network.

## Goals
1. Each K1 runs as an independent chatbot: mic in -> ASR -> LLM (OpenRouter free models) -> TTS -> speaker, plus camera vision.
2. Manual control of the built-in velocity controller is never lost (arbiter, see below).
3. Parallel workstream: load locomotion policies (Isaac Sim/Lab, MuJoCo exports) onto the robot.

## Findings the plan relies on (A2, fw 1.8.1.2, 2026-10-27)
- SDK (`booster_ws/sdk/booster_robotics_sdk/include/booster/robot/ai/`): `AiClient` (StartAiChat 2000, StopAiChat 2001, Speak 2002, Start/StopFaceTracking),
  `LuiApiId` (StartAsr 1000, StopAsr 1001, StartTts 1050, StopTts 1051, SendTtsText 1052), `AsrChunk` idl. Locomotion: `b1_loco_client.hpp` (Move, modes).
- ROS topics on robot: `/LuiApiTopicReq|Resp`, `/LocoApiTopicReq|Resp`, `/booster/audio/*` (init_capture_stream, init_player, get_doa_angle, volume, bluetooth).
- Audio hardware: capture card "NationalChip UAC Speaker" (USB mic array, DOA), playback USB audio device.
- Camera: `/boostercamera/head/rgb` ~27 Hz nv12 544x448; depth ~25 Hz float32 mm. MJPEG helper exists in `booster_ws/tools/k1_camera/depth_sbs_server.py`.
- Robots: A2 192.168.50.215, A3 .230, B1 .105, B2 .62, B3 .248 on robotics_5G. ssh aliases `boosterk1.<name>`.
- Free OpenRouter models (checked 2026-10-27): text `nvidia/nemotron-3.5-lightning:free`, vision `google/gemma-4-31b-it:free`,
  audio-in `thinkingmachines/inkling-small:free`. Only `:free` ids allowed (client rejects others). Free tier is rate limited: retry/backoff, local fallback later.

## Architecture
```
laptop/host (runs this repo)                      K1 (A2, ...)
 brain.py  <- audio.py  <- ASR (robot LUI or OpenRouter audio model) <- mic
   |   \-- vision.py <- camera frames (MJPEG/ROS)
   |-- llm.py -> OpenRouter (free)
   |-- speech out -> LUI TTS / Speak API -> speaker
   '-- actions -> arbiter.py -> B1LocoClient.Move   <- manual teleop/joystick (priority)
```
Runs on the host first (robot has ~4.5 GB free RAM and no internet by default); on-robot packaging later.

## Velocity arbiter (the safety rule)
- Sources: `manual` (joystick/teleop/CLI) and `chat` (LLM tool calls). Manual always wins, is never clamped.
- Chat is clamped (default 0.4 m/s, 0.2 m/s lateral, 0.5 rad/s), expires to zero after 1 s without refresh, is blocked for 2 s after any manual input (even a zero), and `estop()` latches until `release()`.
- Spoken "stop" and any ASR failure map to zero velocity. Robot built-in controller and remote stay usable throughout; chat never changes locomotion mode.
- E-stop = latch arbiter, Move 0, ChangeMode PREPARE (console, fleet dashboard, voice, `k1tools.sh estop`). Release only unlatches.
- P3/P4 split into worktrees `feat/k1-vision` (PLAN_VISION.md) and `feat/k1-tools` (PLAN_TOOLS.md) off contract commit 0890f69.

## Policy-loading workstream (parallel, directory `policy/`, own tests)
1. Export formats: Isaac Lab (`.pt` TorchScript/`policy.onnx`), MuJoCo (`.onnx`), checked against joint order in booster_assets (22 DoF names) and obs/action spec.
2. Loader + validator: onnxruntime, shape/joint-order/limits check, dry run in MuJoCo before robot.
3. Deploy path: custom-mode low-level SDK, 50 Hz loop, 12 leg joints only (per `booster_ws` CLAUDE.md), watchdog back to built-in walking mode. Never auto-run on hardware; needs explicit operator step.
Tests first: spec validator against fixture policies; sim-in-the-loop MuJoCo test; hardware smoke test marked `robot`.

## TDD phases (red -> green)
- P0 (done): scaffold, `.env` (OpenRouter key only), tests for `llm` and `arbiter` written and failing (13 red).
- P1 (done, 13 green; live net test passes on nemotron-3.5-lightning, nemotron-3-super 404s upstream): implement `llm.py` (httpx, `:free` guard, retry on 429/5xx, `LLMError`) and `arbiter.py` until green. Live test `-m net`.
- P2 (fw 1.8.x only, local ASR/TTS on the robot, no cloud): `audio.py` (VAD segmenter, sentence-by-sentence speak, 16 tests), `robot_audio.py` (arecord -> faster-whisper tiny.en (CPU, ~1 s per 5 s audio vs 3.6 s for base.en); Kokoro-82M fp32 on the robot GPU via the system onnxruntime-gpu, RTF 0.19 on A3 vs 2.8 on CPU; aplay), `robot/run.sh` (frees the mic via containerd `POST :20001/service/kill|boot {"service":"audio"}`; killing the process only makes containerd respawn it in <2 s), `scripts/k1voice.sh start|stop|smoke|status|restore <robot>`, `scripts/k1voice_install.sh <robots>`. Open: live mic level (peak rms 728-1390 vs threshold 500, no utterance heard on A1), `-m robot` test, latency tuning.
- P3: `vision.py`: frame grab, JPEG -> vision model, describe scene test with fixture image, then `-m robot`.
- P4: `brain.py`: conversation loop, tool calls (`move`, `stop`, `look`, `say`), persona prompt; tests with scripted LLM.
- P5: multi-robot: one brain per robot, namespaced by `robot_ns`; independent state and history.
- P6: policy workstream P1-P3 above.

## Run tests
`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest --with respx --with httpx --with python-dotenv pytest` (the env vars keep ROS pytest plugins out).
Live: `... pytest -m net` / `-m robot` (needs `K1_IP`).

## Open questions
- ASR/TTS path: robot LUI services (needs verifying they work offline) vs OpenRouter audio model vs local whisper on the host.
- Whether 1.8.1.2 AI chat (`StartAiChat`) talks to a Booster cloud. Avoid it; we use our own LLM loop. Check network egress before enabling.
- Free models may log prompts; no private audio/video of people without consent.
- Keys: only `OPENROUTER_API_KEY` was copied. Other keys in intent-sim `.env` (NIM, OpenAI, Gemini) were left there on purpose.

## Next (see HANDOFF.md for state, issues, details)
- Tool calling (`run_python`, where it runs is undecided), barge-in/interruption, OpenAI Realtime API spike (paid, later), per-robot calibration, ASR noise hallucinations.
