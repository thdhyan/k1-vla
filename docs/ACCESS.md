# How to access the K1 robots

## SSH
- Key `~/.ssh/booster_k1` (ed25519, no passphrase; never commit or copy it). User `booster`. Passwordless sudo on the robots.
- `ssh boosterk1.a2` (aliases a1..b3; template in `ssh_config.example`). Or `ssh -i ~/.ssh/booster_k1 booster@192.168.50.215`.
- Laptop must be on the robots' L2 net (`enp2s0` 192.168.50.x). Check: `ping 192.168.50.215`.
- Long commands that change the robot's network must be `nohup`-detached, the ssh session drops.

## ROS 2 on the robot
`source /opt/ros/humble/setup.bash; source /opt/booster/ros2/setup.bash`, `ROS_DOMAIN_ID=0`. Useful topics:
`/LocoApiTopicReq|Resp` (locomotion RPC), `/LuiApiTopicReq|Resp` (ASR/TTS), `/booster/audio/*`, `/boostercamera/head/{rgb,depth}`.
The laptop is Jazzy; DDS over wifi is flaky (multicast filtered). Prefer running ROS code on the robot and talking HTTP/websocket to it.
Never mix Isaac Sim's ROS with system ROS in one shell.

## SDK
`booster_ws/sdk/booster_robotics_sdk` (read-only; C++ and python). `B1LocoClient` (Move, modes), `AiClient` (Speak, StartAiChat), `LuiApiId` (ASR/TTS).
Robot must be on the same DDS domain/subnet as the host.

## Studio websocket `ws://<ip>:51111`
Protobuf, server push only. field1=10001 camera JPEG (RGB, ~10 fps), field1=10000 status (battery, joints, robot info). No depth, no audio. Notes: `booster_ws/docs/booster_websocket_studio_cli_findings_2026-10-27.md`.

## Camera + depth over HTTP
`tools/depth_sbs_server.py` (run on the robot, see skill `k1-camera-stream`). `http://<ip>:8090/` side-by-side, `/snapshot.jpg`, `/stats`.

## Do NOT
- Run `booster-cli upgrade` (even `-h`): it downloads and installs the latest firmware, cannot be pinned. Read versions with `grep Version /opt/booster/version.txt | tail -1`.
- Run `booster-cli upload_log` (sends logs to Booster cloud).
- Auto-run learned policies on hardware, or let chat change locomotion mode. Keep the robot supported/on a gantry for any low-level test.
- Use `StartAiChat` without checking network egress (may use Booster cloud).

## Battery / status quickly
`uv run booster_ws/tools/fleet_dashboard/dashboard.py --port 8080` (all robots, online + battery).
