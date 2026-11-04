#!/usr/bin/env bash
# Self-hosted OpenAI-compatible models for the K1 brain (vLLM in docker on a GPU box). Normally driven by the LLM panel
# (laptop: PYTHONPATH=src uv run python -m k1_chat.llm_panel --ssh-host dl --gpu 3), which adds the ssh tunnel, logging and buttons.
#   scripts/serve_llm.sh up <ssh-host> <gpu> chat|vision    start container (chat: Bolt E4B :8000, vision: Cosmos-Reason2 :8001)
#   scripts/serve_llm.sh down <ssh-host> chat|vision|all    stop it (frees the GPU)
#   scripts/serve_llm.sh status <ssh-host> [gpu]            containers + GPU memory
#   scripts/serve_llm.sh test [base-url]                    tool-call test (pytest -m onprem), default the panel
# Override models: CHAT_MODEL=..., VISION_MODEL=nvidia/Cosmos-Reason2-8B. Image pinned: vllm-openai:v0.23.0 (has gemma4 + qwen3 parsers).
# Flags from the model cards (huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16, nvidia/Cosmos-Reason2-2B).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=vllm/vllm-openai:v0.23.0
CHAT_MODEL=${CHAT_MODEL:-nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16}
VISION_MODEL=${VISION_MODEL:-nvidia/Cosmos-Reason2-2B}
case "${1:-}" in
up)
    host=${2:?ssh host}; gpu=${3:?gpu index}; role=${4:?chat|vision}
    case $role in
    chat) port=8000; args="--model $CHAT_MODEL --served-model-name nemotron-bolt --gpu-memory-utilization 0.45 --max-model-len 16384
            --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --limit-mm-per-prompt {\"image\":0,\"video\":0,\"audio\":0}" ;;
    vision) port=8001; args="--model $VISION_MODEL --served-model-name cosmos-reason2 --gpu-memory-utilization 0.25 --max-model-len 16384
            --limit-mm-per-prompt {\"image\":1,\"video\":0}" ;;  # no reasoning parser: it answers without <think>, parser moved all text to "reasoning"
    *) echo "role: chat|vision" >&2; exit 2 ;;
    esac
    echo "starting k1-llm-$role on $host gpu $gpu port $port"
    # shellcheck disable=SC2029
    ssh "$host" "docker rm -f k1-llm-$role >/dev/null 2>&1; docker run -d --name k1-llm-$role --gpus device=$gpu --ipc=host \
      -p 127.0.0.1:$port:8000 -v \$HOME/.cache/huggingface:/root/.cache/huggingface $IMAGE --dtype bfloat16 $(printf '%q ' $args)" &&
    echo "loading... (watch the panel status, or: ssh $host docker logs -f k1-llm-$role)" ;;
down)
    host=${2:?ssh host}; role=${3:-all}
    names=k1-llm-$role; [ "$role" = all ] && names="k1-llm-chat k1-llm-vision"
    # shellcheck disable=SC2029
    ssh "$host" "for n in $names; do docker rm -f \$n 2>/dev/null && echo removed \$n || echo \$n not running; done" ;;
status)
    host=${2:?ssh host}; gpu=${3:-}
    # shellcheck disable=SC2029
    ssh -o ConnectTimeout=8 "$host" "docker ps -a --filter name=k1-llm --format '{{.Names}}: {{.Status}}';
      nvidia-smi ${gpu:+-i $gpu} --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/gpu /';
      df -h --output=avail \$HOME/.cache/huggingface | tail -1 | sed 's/^ */disk free: /'" ;;
test)
    cd "$root"
    K1_CHAT_BASE_URL=${2:-${K1_CHAT_BASE_URL:-http://192.168.50.250:8000/v1}} K1_CHAT_MODEL=nemotron-bolt \
      env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest --with respx --with httpx --with python-dotenv \
      pytest -q -s -m onprem tests/test_llm.py ;;
*) sed -n 2,10p "$0"; exit 2 ;;
esac
