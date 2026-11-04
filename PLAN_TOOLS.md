# K1 robot tools: plan + morning checklist (branch `feat/k1-tools`, worktree `~/Projects/wt-k1-tools`)

Base: contract commit `0890f69` on `feat/k1-chatbot` (llm `complete`/`image_part`, `backend.py`, `tools.py`, `events.py`, `safety.py`, `console.py`).
Sibling: `feat/k1-vision` (`~/Projects/wt-k1-vision`, `look` tool). Merge both into `feat/k1-chatbot` after the robot checks below.

## Rule change (vs HANDOFF "chat must never move the robot")
Chat may move the robot, but only through these tools and the VelocityArbiter (manual wins, chat clamped 0.4/0.2/0.5, 1 s expiry, e-stop latches).
Tools never change mode. The only mode change is e-stop -> PREPARE. `get_up` left out of v1 on purpose.

## API (narrow tools, one backend seam `RobotBackend.call(api_id, param) -> (code, resp)`)
| tool | args | Booster API |
|---|---|---|
| walk | distance_m -0.5..1.5, speed slow 0.15 / normal 0.3 | Move 2001 at 10 Hz via arbiter, then Move 0 |
| turn | angle_deg ±180 (left +) | Move 2001 vyaw 0.4 |
| walk_and_turn | vx -0.2..0.4, vyaw ±0.5, duration_s 0.5..5 | Move 2001 |
| stop | - | Move 0 |
| set_head | yaw_deg ±45 (left +), pitch_deg ±25 (down +, UNVERIFIED) | RotateHead 2004 |
| wave | seconds 1..5 | WaveHand 2005 open, sleep, close |
| handshake | seconds 2..10 | Handshake 2015 open, sleep, close |
| look (vision branch) | question | Studio ws :51111 JPEG -> VLM |

Every tool: refuses while e-stopped, args clamped/enum-checked, never raises (errors go back to the LLM as text), logs tool_call/tool_result.
Moves need WALKING mode (code -2 sim / non-zero real -> "not in walking mode, use the remote").

## E-stop (PREPARE), 4 paths, physical remote stays primary
1. Robot console `http://<robot>:8091/` STOP button (POST /api/estop).
2. Fleet dashboard (laptop, `booster_ws/tools/fleet_dashboard/dashboard.py`) Agent column STOP -> proxies to 1.
3. Spoken "stop"/"halt"/"freeze": handled inline, no LLM.
4. `scripts/k1tools.sh estop <robot>`: console first, then ssh `python -m k1_chat.booster_robot prepare` (works if brain is dead).
Order: latch arbiter -> Move 0 -> ChangeMode PREPARE. Release only unlatches (never changes mode).

## Console (per robot, :8091)
STOP / release, state line (mode, twist, estopped), last VLM frame, event list (heard, llm_request, llm_reply, tool_call, tool_result, frame, vlm_reply, said, estop), 500 ms poll.

## Run
- Laptop sim, typed: `PYTHONPATH=src uv run python -m k1_chat.brain --name A2 --backend sim` -> http://127.0.0.1:8091/
- Robot: `scripts/k1voice.sh start <robot> agent` (`.env`: K1_CHAT_MODEL, optional K1_VISION_MODEL, K1_BACKEND=sim|ros)
- Tests: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest --with respx --with httpx --with python-dotenv pytest -q` (103 pass). `-m robot` needs `K1_IP` + ROS on robot.

## Morning checklist (robots)
0. Laptop NAT up (CLAUDE.md network note), fleet dashboard: `uv run tools/fleet_dashboard/dashboard.py --host 0.0.0.0` in booster_ws.
1. `scripts/k1tools.sh check <robot>`: read-only. Topic types of /LocoApiTopicReq|Resp, Studio frame, rclpy in venv, GetMode, :8091 free.
   Record the message type name in this file.
2. Dry run on the robot with fake body: `K1_BACKEND=sim scripts/k1voice.sh start <robot> agent`. Talk; watch console. Nothing moves.
3. Robot on gantry or held by a person, remote in hand, PREPARE then WALKING via remote. `K1_BACKEND=ros ... start <robot> agent`.
4. Test each STOP path (console, dashboard, voice, `k1tools.sh estop`) -> robot goes PREPARE each time. Then release.
5. Verify, write results here:
   - [ ] message type + response uuid matching (GetMode returns a mode)
   - [ ] status codes: 0 ok; code when Move sent outside WALKING
   - [ ] set_head pitch sign (+ = down?) and yaw sign (+ = left?)
   - [ ] remote vs Move: does remote stick override while a tool drives? (arbiter only knows our own manual source)
   - [ ] WaveHand HandIndex (right=1?), Handshake open/close
   - [ ] walk 0.5 m actually ~0.5 m (open loop, timed)
6. Note every smoke/test on Slides deck `1l00na5AEsAdHfaahgO_VbWNqkhfFlzRfNjzxCtqnZhQ` / Drive folder `19TTmMfsWvI-BS3srMDhxWrB9MjlGJtU5`.

## Open
- Limits: walk ≤1.5 m, turn ≤180°, chat speed ≤0.3 m/s; raise only after step 5.
- Remote override: if Move from chat masks the stick, add a remote-input listener feeding `arbiter.manual`.
