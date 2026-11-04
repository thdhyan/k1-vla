# k1-vla

k1_chat voice/VLM agent, tools and YOLO node.

| | |
|---|---|
| Layer | **L2**: Domain (no cross-imports) |
| May import | [k1-common](https://github.com/thdhyan/k1-common), [k1-utils](https://github.com/thdhyan/k1-utils), [k1-runtime](https://github.com/thdhyan/k1-runtime) |
| Files / LOC | 71 files · 5,976 lines (py, sh, cpp) |
| Part of | [k1-workspace](https://github.com/thdhyan/k1-workspace) (architecture and layering rules live there) |

## Install

```bash
uv add git+https://github.com/thdhyan/k1-vla
# or, from a checkout
uv sync
```

## Layout

```
HANDOFF.md
PLAN.md
PLAN_TOOLS.md
README.md
docs/  (6 files)
  ACCESS.md
  AUDIO_PIPELINE.md
  DEMO_RUNBOOK.md
  FLEET.md
  README.md
  ssh_config.example
fleet.json
pyproject.toml
robot/  (1 files)
  run.sh
scripts/  (7 files)
  README.md
  demo.sh
  k1tools.sh
  k1voice.sh
  k1voice_install.sh
  probe_vl_tools.py
  serve_llm.sh
src/  (31 files)
  k1_chat/  (31 files)
tests/  (24 files)
  README.md
  test_arbiter.py
  test_asr_adapter.py
  test_asr_server.py
  test_audio.py
  test_backend.py
  test_booster_robot.py
  test_brain.py
  test_console.py
  test_conversation.py
  test_conversation_tools.py
  test_debug.py
  test_events_safety.py
  test_fleet.py
  test_kokoro_mem_limit.py
  test_llm.py
  test_llm_panel.py
  test_llm_tools.py
  test_modes.py
  test_robot_tools.py
  test_script_tools.py
  test_tools.py
  test_vision.py
  test_wake.py
tools/  (1 files)
  depth_sbs_server.py
```

## Dependencies

- Declared sibling libraries: k1-common, k1-utils, k1-runtime (see `pyproject.toml`, `[tool.uv.sources]`).
- PyPI: httpx, numpy, onnxruntime, opencv-python, pytest, websockets.
- Provided by the environment (ROS 2, Isaac, mjlab, robot SDK) or unresolved: `dotenv`, `faster_whisper`, `geometry_msgs`, `kokoro_onnx`, `rclpy`, `respx`, `rosidl_runtime_py`, `sensor_msgs`, `std_msgs`, `ultralytics`.

## Hard-coded paths into other repos

These files still name paths that now live in another repo (hard-coded path strings, not imports). Pass them as arguments / config when you touch the file:

| File | Refers to | Owner |
|---|---|---|
| `PLAN.md` | `tools/` | k1-web-console |
| `PLAN_TOOLS.md` | `tools/` | k1-web-console |
| `docs/ACCESS.md` | `tools/` | k1-web-console |
| `docs/FLEET.md` | `tools/` | k1-web-console |

## Origin

Extracted from `booster_ws` (branch `dev/push-training-wip`, working tree of 2026-11-03; k1_chat from `wt-k1-tools`) by
`tools/build_workspace.py` in k1-workspace. Old path to new path for every file: [`docs/migration_map.tsv`](https://github.com/thdhyan/k1-workspace/blob/main/docs/migration_map.tsv) (filter on `k1-vla`).
History was not carried over: the working tree was copied as is, including previously untracked files.
