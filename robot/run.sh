#!/usr/bin/env bash
# Runs on the K1 (fw 1.8.x). Frees the mic by stopping the robot's "audio" service (supervised by containerd,
# so killing the process just respawns it; use its API), runs k1_chat.robot_audio <args>, and ALWAYS starts
# the audio service again on exit (normal end, SIGTERM, crash, ssh drop).
#   run.sh smoke "sentence" | run.sh listen | run.sh asr-file x.wav
#   run.sh --status | run.sh --restore         audio service state / make sure it is running
API=http://127.0.0.1:20001/service
K1=$HOME/k1_chat
PIDFILE=$K1/run.pid
child=""

api() { curl -sm8 -X POST -H "content-type: application/json" -d '{"service":"audio"}' "$API/$1"; }
audio_up() { api status | grep -q '"stage":"ONLINE"'; }

wait_state() {  # wait_state up|down
    for _ in 1 2 3 4 5 6 7 8; do
        if audio_up; then [ "$1" = up ] && return 0; else [ "$1" = down ] && return 0; fi
        sleep 1
    done
    return 1
}

case ${1:-} in
    --status)  audio_up && echo "audio service: running" || echo "audio service: stopped"; exit 0 ;;
    --restore) audio_up && { echo "audio service already running"; exit 0; }
               api boot >/dev/null; wait_state up && echo "audio service restored" || { echo "WARNING: audio service still down" >&2; exit 1; }
               exit 0 ;;
esac

restore() {
    [ -n "$child" ] && kill "$child" 2>/dev/null && wait "$child" 2>/dev/null
    if audio_up; then echo "audio service running"
    else api boot >/dev/null; wait_state up && echo "audio service restored" || echo "WARNING: audio service NOT running" >&2; fi
}

# Preflight: change nothing on the robot unless we can run AND restore.
api status | grep -q '"success":true' || { echo "containerd API not answering on :20001 (fw 1.8.x needed)" >&2; exit 2; }
[ -x "$K1/venv/bin/python" ] || { echo "no ~/k1_chat/venv on this robot: run scripts/k1voice_install.sh" >&2; exit 2; }

# Nemo ASR first: its ~60 s model load plus our start does not fit in RAM at once (OOM-killed it on A2)
if [ "${K1_CHAT_ASR:-nemo}" = nemo ] && systemctl is-enabled -q nemo-asrd 2>/dev/null; then
    echo "waiting for nemo-asrd to be ready"
    for _ in $(seq 90); do curl -s -m2 http://127.0.0.1:8090/health | grep -q '"ready": 1' && break; sleep 2; done
fi

echo $$ > "$PIDFILE"
trap 'restore; rm -f "$PIDFILE"' EXIT
trap 'exit 143' TERM INT HUP

api kill >/dev/null
wait_state down || { echo "audio service would not stop" >&2; exit 1; }

cd "$K1" || exit 1
if [ "${1:-}" = agent ] && [ "${K1_BACKEND:-ros}" = ros ]; then  # robot tools talk loco RPC over ROS 2
    set +u; source /opt/ros/humble/setup.bash; source /opt/booster/ros2/setup.bash; export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
    # booster_interface (BatteryState) ships only inside a vendor agent bundle; optional, for the robot_status tool
    for b in /opt/booster/booster_agent_data/data/agents/extract/*/agent/booster_interface/share/booster_interface/local_setup.bash; do
        [ -f "$b" ] && { source "$b"; break; }
    done
fi
K1_CHAT_ASR="${K1_CHAT_ASR:-nemo}" HF_HUB_OFFLINE=1 PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH} venv/bin/python -m k1_chat.robot_audio "$@" &
child=$!
wait "$child"
