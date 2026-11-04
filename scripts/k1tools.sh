#!/usr/bin/env bash
# Laptop side: robot tools + console. Morning order: check -> (robot on gantry/held) agent dry run -> agent live.
#   k1tools.sh check   <robot>   read-only: ROS loco topics + message type, camera ws :51111, rclpy in venv, GetMode
#   k1tools.sh estop   <robot>   console POST /api/estop; if the brain is down: ssh fallback (zero Move + PREPARE)
#   k1tools.sh mode    <robot>   read the locomotion mode (read-only)
#   k1tools.sh console <robot>   print the console URL
# Start/stop the voice agent with: scripts/k1voice.sh start <robot> agent | stop <robot>
set -euo pipefail
cmd=${1:?usage: k1tools.sh check|estop|mode|console <robot>}
name=${2:?robot name, e.g. a2}
host=boosterk1.${name,,}
root=$(cd "$(dirname "$0")/.." && pwd)
ip=$(python3 -c "import json,sys;print(json.load(open('$root/src/k1_chat/fleet.json'))['robots']['${name^^}']['ip'] or '')")
[ -n "$ip" ] || { echo "no IP for $name in fleet.json" >&2; exit 2; }
ROS='source /opt/ros/humble/setup.bash; source /opt/booster/ros2/setup.bash; export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}'
PY='cd ~/k1_chat && PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH} venv/bin/python'

sync() {
    ssh -o ConnectTimeout=5 "$host" 'mkdir -p ~/k1_chat/src/k1_chat'
    rsync -aq --delete --exclude __pycache__ "$root/src/k1_chat/" "$host:k1_chat/src/k1_chat/"
}

case $cmd in
  check)
    sync
    echo "== loco RPC topics (need /LocoApiTopicReq + Resp)"
    ssh "$host" "$ROS; for t in /LocoApiTopicReq /LocoApiTopicResp; do echo \"\$t: \$(ros2 topic type \$t 2>&1)\"; done;
                 T=\$(ros2 topic type /LocoApiTopicReq 2>/dev/null) && ros2 interface show \$T"
    echo "== camera: Studio websocket from laptop"
    uv run --quiet --with websockets python -I -c "
import sys; sys.path.insert(0, '$root/src')
from k1_chat.frames import StudioFrames
j = StudioFrames('$ip').grab(); print('frame bytes:', len(j) if j else None)" 2>&1 || echo "frames.py missing (vision branch not merged yet)"
    echo "== rclpy inside ~/k1_chat/venv"
    ssh "$host" "$ROS; $PY -c 'import rclpy, rosidl_runtime_py; print(\"rclpy ok\")'"
    echo "== GetMode (read-only RPC)"
    ssh "$host" "$ROS; $PY -m k1_chat.booster_robot get-mode"
    echo "== console port free?"
    ssh "$host" "ss -ltn | grep -q ':8091 ' && echo ':8091 in use (agent running?)' || echo ':8091 free'"
    ;;
  estop)
    if curl -sf -m 2 -X POST "http://$ip:8091/api/estop"; then echo; echo "e-stop via console OK"
    else
        echo "console unreachable, ssh fallback" >&2
        sync
        ssh -o ConnectTimeout=4 "$host" "$ROS; $PY -m k1_chat.booster_robot prepare"
    fi
    ;;
  mode)    sync; ssh "$host" "$ROS; $PY -m k1_chat.booster_robot get-mode" ;;
  console) echo "http://$ip:8091/" ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
