#!/usr/bin/env bash
# One command for the talking-robot demo (A1 = "Apollo"). Run from the laptop, any directory. Runbook: docs/DEMO_RUNBOOK.md
#   demo.sh up     [robot]   laptop NAT + GPU ASR server, robot YOLO/depth service, voice agent + console
#   demo.sh down   [robot]   stop agent + detect on the robot (robot audio service comes back); --all also stops laptop ASR
#   demo.sh status [robot]   health of every piece
#   demo.sh log    [robot]   follow what the robot hears / ignores / says (Ctrl-C leaves it running)
# Env: K1_LAPTOP_IP (default 192.168.50.250, the robots' NAT gateway), K1_WAKE_WORDS / K1_CALL_NAME to override the name.
set -euo pipefail
cmd=${1:-status}
robot=${2:-a1}
[ "${2:-}" = --all ] && robot=a1
root=$(cd "$(dirname "$0")/.." && pwd)
host=boosterk1.${robot,,}
laptop=${K1_LAPTOP_IP:-192.168.50.250}
asr_log=$root/audio_dumps/asr_server.log
robot_ip=$(ssh -G "$host" | awk '/^hostname /{print $2}')

ok()   { printf '  \e[32mOK\e[0m   %s\n' "$*"; }
bad()  { printf '  \e[31mDOWN\e[0m %s\n' "$*"; }
asr_up()    { curl -sf -m 2 "http://127.0.0.1:8090/health" >/dev/null; }
detect_up() { curl -sf -m 2 "http://$robot_ip:8092/health" >/dev/null; }
agent_up()  { curl -sf -m 2 -o /dev/null "http://$robot_ip:8091/"; }

start_asr() {
    asr_up && { ok "laptop ASR already running (:8090)"; return; }
    mkdir -p "$(dirname "$asr_log")"
    echo "starting laptop GPU ASR (faster-whisper large-v3-turbo), first start downloads ~1.6 GB..."
    (cd "$root" && PYTHONPATH=src setsid nohup uv run --quiet --with faster-whisper --with nvidia-cublas-cu12 \
        --with "nvidia-cudnn-cu12==9.*" python -u -m k1_chat.asr_server --model large-v3-turbo \
        > "$asr_log" 2>&1 < /dev/null &)
    for _ in $(seq 120); do asr_up && { ok "laptop ASR up"; return; }; sleep 2; done
    bad "laptop ASR did not come up, see $asr_log"; exit 1
}

status() {
    echo "== $robot ($host, $robot_ip)"
    ssh -o ConnectTimeout=5 "$host" true 2>/dev/null && ok "ssh" || { bad "ssh: robot off or not on the 5G network"; return; }
    asr_up && ok "laptop ASR :8090" || bad "laptop ASR :8090 (demo.sh up starts it)"
    detect_up && ok "robot YOLO/depth :8092 (http://$robot_ip:8092/detections)" || bad "robot YOLO/depth :8092 (forward walking refused)"
    agent_up && ok "agent console http://$robot_ip:8091/" || bad "agent console :8091"
    ssh -T "$host" 'free -m | awk "/Mem:/{print \"  RAM  \" \$7 \" MB available\"}"; grep -c "Killed process" <(journalctl -k -b --no-pager 2>/dev/null) | sed "s/^/  OOM  kills this boot: /"' || true
}

case $cmd in
  up)
    ssh -o ConnectTimeout=5 "$host" true || { echo "cannot ssh $host: robot on and on the 5G network?" >&2; exit 1; }
    start_asr
    ssh "$host" '[ -f ~/k1_chat/models/yolo11n.onnx ] || cp /opt/booster/BoosterFaceDetection/src/detection/yolo11n.onnx ~/k1_chat/models/'
    bash "$root/scripts/k1voice.sh" detect "$robot" start
    agent_up && bash "$root/scripts/k1voice.sh" stop "$robot" >/dev/null   # fresh agent picks up the synced code
    K1_BACKEND=ros K1_CHAT_ASR=nemo K1_NEMOTRON_URL="http://$laptop:8090/transcribe" \
        bash "$root/scripts/k1voice.sh" start "$robot" agent
    for _ in $(seq 30); do agent_up && break; sleep 2; done
    echo; status
    echo; echo "Console: http://$robot_ip:8091/   (STOP button, mode buttons, wake words, live camera)"
    echo "Say: \"Apollo, ...\"  -- every command starts with the name; \"stop\" always works." ;;
  down)
    bash "$root/scripts/k1voice.sh" stop "$robot" || true
    bash "$root/scripts/k1voice.sh" detect "$robot" stop || true
    if [[ " $* " == *" --all "* ]]; then pkill -f "[k]1_chat.asr_server" && echo "laptop ASR stopped" || true; fi ;;
  status) status ;;
  log) ssh -t "$host" 'tail -n 30 -f ~/k1_chat/voice.log | grep --line-buffered -E "HEARD|IGNORED|SAID|listening for|skipped|rror"' ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac
