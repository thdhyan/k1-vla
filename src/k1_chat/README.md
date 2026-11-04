# src/k1_chat

Part of [k1-vla](https://github.com/thdhyan/k1-vla).

| File | What it does |
|---|---|
| `__init__.py` |  |
| `arbiter.py` | Velocity arbiter: manual control always beats chat. tests/test_arbiter.py defines the contract. |
| `asr_server.py` | Laptop GPU ASR with the nemo-asrd HTTP API, so robots offload speech recognition and keep their RAM. |
| `audio.py` | Local voice pipeline: mic frames -> VAD segmenter -> ASR text, and text -> TTS wav -> player. |
| `backend.py` | Robot backend: one call(api_id, param) seam, mirroring B1LocoClient::SendApiRequest. tests/test_backend.py. |
| `booster_robot.py` | Real K1 backend: loco RPC over ROS 2 (/LocoApiTopicReq -> /LocoApiTopicResp), the same DDS topics B1LocoClient uses. Runs ON the robot (humb |
| `brain.py` | Brain: one robot = backend + arbiter + safety + tools + conversation + web console. tests/test_brain.py. |
| `console.py` | Per-robot web console, served by the brain on the robot (default :8091). tests/test_console.py. |
| `conversation.py` | Voice conversation: transcript in, short spoken reply out. With a tool Registry the model can also call tools (robot_tools.py: every motion  |
| `debug.py` | Health snapshot for the console's System panel (GET /api/debug). tests/test_debug.py. |
| `detect_server.py` | Perception service on the robot: every YOLO detection + an obstacle check from depth, over HTTP. tests/test_detect_server.py. |
| `events.py` | Thread-safe ring buffer of brain events (heard, llm_request, llm_reply, tool_call, tool_result, said, estop, ...) for the web console, plus  |
| `fleet.py` | Fleet registry loaded from fleet.json (single source of truth for robot IPs). |
| `frames.py` | Camera frame sources (JPEG bytes). tests/test_vision.py. |
| `llm.py` | OpenRouter chat client (free models only) or a self-hosted OpenAI-compatible server (vLLM, llama.cpp). tests/test_llm.py and tests/test_llm_ |
| `llm_panel.py` | LLM panel on the laptop: logging proxy robots -> self-hosted vLLM on a GPU box, plus load/unload buttons so the GPU is only held while we us |
| `modes.py` | Locomotion mode FSM shared by the chat tools and the e-stop. tests/test_modes.py. |
| `perception.py` | Client for detect_server (YOLO + depth obstacle, :8092 on the robot). tests/test_perception.py. |
| `robot_audio.py` | On-robot audio: arecord -> Segmenter -> faster-whisper, Kokoro (GPU) -> aplay. Run via robot/run.sh (needs the mic free). |
| `robot_tools.py` | Robot tools for the LLM: narrow flat tools over one Loco backend. tests/test_robot_tools.py. |
| `safety.py` | E-stop = latch the arbiter, send zero velocity, then ModeFSM.interrupt(): WALKING/CUSTOM/SOCCER -> PREPARE (stands still in place); DAMPING  |
| `script_tools.py` | `macro` and `run_script`: let the model chain robot tools in one call. tests/test_script_tools.py. |
| `system_prompt.md` |  |
| `tools.py` | LLM tool registry: narrow flat tools -> OpenAI function schemas; run() never raises, errors go back to the model as text. Numbers are clampe |
| `vision.py` | `look` tool: grab one camera frame, ask the vision model, return its text to the chat model. tests/test_vision.py. |
| `yolo_node.py` | YOLO on the K1's left ZED eye: ``/{ns}/zed/left/image_raw`` -> ``/{ns}/yolo/image_raw`` (annotated rgb8, same stamp; the headset's YOLO pane |
