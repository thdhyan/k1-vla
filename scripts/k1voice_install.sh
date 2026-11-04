#!/usr/bin/env bash
# Install the local voice stack on K1s (fw 1.8.x): faster-whisper tiny.en (CPU, 3x faster than base.en) + Kokoro-82M TTS on the robot GPU
# via the system onnxruntime-gpu. Idempotent. Needs internet on the robot (laptop NAT); adds a non-persistent
# default route via the laptop if the robot has none (DHCP robots default to the internet-less router).
#   k1voice_install.sh a1 a2 a3 b1 b2 b3
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: k1voice_install.sh a1 a2 a3 b1 b2 b3" >&2; exit 2; }
REL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0

install_one() {
    local host=boosterk1.${1,,}
    ssh -o ConnectTimeout=6 "$host" "REL=$REL bash -s" <<'REMOTE' 2>&1 | sed "s/^/[$1] /"
set -u
curl -sm6 -o /dev/null https://pypi.org || {
    sudo ip route add default via 192.168.50.250 dev wlP1p1s0 metric 100 2>/dev/null
    sudo resolvectl dns wlP1p1s0 8.8.8.8 1.1.1.1 2>/dev/null   # DHCP robots get a DNS that answers 10.0.0.1
    curl -sm6 -o /dev/null https://pypi.org || { echo "NO INTERNET (is the laptop NAT on?)"; exit 1; }
}
mkdir -p ~/k1_chat/models && cd ~/k1_chat
# system site-packages = the robot's onnxruntime-gpu (CUDA); it is built for NumPy 1.x, so pin numpy<2
if [ ! -f venv/.k1gpu ]; then
    rm -rf venv && python3 -m venv --system-site-packages venv || { echo "venv failed"; exit 1; }
fi
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q faster-whisper kokoro-onnx httpx websockets || { echo "pip failed"; exit 1; }
venv/bin/pip uninstall -y -q onnxruntime >/dev/null 2>&1   # CPU wheel would shadow the system GPU one
venv/bin/pip install -q "numpy<2" 2>&1 | grep -v "requires numpy" ; touch venv/.k1gpu
cd models
for f in kokoro-v1.0.onnx voices-v1.0.bin; do [ -s $f ] || curl -sL -C - -O $REL/$f || { echo "download $f failed"; exit 1; }; done
cd ..
HF_HUB_DISABLE_PROGRESS_BARS=1 venv/bin/python -c 'from faster_whisper import WhisperModel; WhisperModel("tiny.en", device="cpu", compute_type="int8")' 2>&1 | grep -v Warning
venv/bin/python - <<'PY' 2>&1 | grep -v "W:onnx" | tail -3
import onnxruntime as o
p = o.get_available_providers()
assert "CUDAExecutionProvider" in p, f"no CUDA provider: {p}"
print("CUDA provider ok, ort", o.__version__)
PY
echo "INSTALL DONE: $(grep Version /opt/booster/version.txt | tail -1)"
REMOTE
}

for r in "$@"; do install_one "$r" & done
wait
