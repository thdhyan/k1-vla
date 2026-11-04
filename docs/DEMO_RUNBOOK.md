# Talking robot demo: runbook (A1 "Apollo")

Booster K1 **A1** listens, talks (British voice), sees (YOLO and depth on the robot, VLM in the cloud) and moves through
LLM tool calls. It answers only to **"Apollo"**. Everything starts from the laptop with one script.

## What runs where

| Piece | Where | Port | Started by |
|---|---|---|---|
| Speech-to-text (faster-whisper large-v3-turbo, GPU) | laptop | :8090 | `demo.sh up` |
| Internet for the robot (NAT through laptop's eduroam) | laptop | n/a | `demo.sh up` (lost on laptop reboot) |
| YOLO + depth obstacle service | robot | :8092 | `demo.sh up` |
| Voice agent: mic, LLM + tools, Kokoro TTS, web console | robot | :8091 | `demo.sh up` |
| Chat LLM + VLM (free OpenRouter models) | cloud | n/a | key from `.env`, sent over ssh, never stored on the robot |

## Before people arrive (5 min)

1. **Battery above 40%.** The robot falls when the battery runs low (it happened at about 15%). Battery % is in the console Events (`state` lines); under 20% the robot says so and avoids walking.
2. Power on the robot, wait about 1 min, and check that the laptop is on the robotics 5G network (`ssh boosterk1.a1 true`).
3. Stand the robot up with the remote (damping → prepare), with clear floor of about 2 m around it.
4. On the laptop:
   ```bash
   cd ~/Projects/wt-k1-tools
   bash scripts/demo.sh up        # about 1 min; ends with a status table, all OK
   ```
5. Open the console **http://192.168.50.82:8091/** on the laptop or a phone, and keep it visible to the operator.
6. Press **walking mode** in the console, or let the robot do it: it switches itself when asked to move.
7. Test phrase: *"Apollo, what do you see?"* It should answer in one or two short sentences.

## Talking to it

- **Every command starts with the name:** "Apollo, turn left a little." Other room talk is ignored ("ignored" lines in the console).
- **"Apollo"** on its own gives "Yes?", and the next sentence is heard without the name.
- **When it asks you a question** it hears your next sentence without the name (10 s). After that you need the name again.
- **"Stop"** always works without the name: the robot stands still (prepare) and chat motion is locked until the operator presses **chat motion ON** in the console.

Good demo lines:

| Say | What happens |
|---|---|
| "Apollo, what do you see?" | quick YOLO answer: objects, angles, distances |
| "Apollo, describe the scene in detail" | camera frame to the VLM |
| "Apollo, turn left 30 degrees" / "walk forward half a metre" | moves; stops by itself about 0.5 m before obstacles |
| "Apollo, look for a person" | turns head and body to search |
| "Apollo, follow me" | follows the nearest person for 30 s, keeps them centred, stops if lost for 2 s |
| "Apollo, wave at them" / "shake my hand" / "do a dab" | gestures |
| "Apollo, what is MnRI?" / "who are you?" | chat with lab background notes |

## Safety (operator)

- **Console STOP** (big red button) or saying **"stop"**: chat motion latches off and the robot goes to prepare (stands still). Press **chat motion ON** to allow motion again.
- **chat motion OFF**: the robot keeps talking but cannot move (use it in crowds).
- **Remote control always wins.** If anything looks wrong, use the remote's damping or prepare.
- Speed is capped at 0.4 m/s. Forward walking is refused if the YOLO/depth service is down, so it never walks blind.
- Keep the area in front of it clear. Depth sees chairs and walls, but not low cables or glass.

## Afterwards

```bash
bash scripts/demo.sh down          # robot back to its own audio service; add --all to also stop laptop ASR
```
Put the robot in damping with the remote, then charge it.

## Other commands

```bash
bash scripts/demo.sh status        # health of every piece, robot RAM, OOM kills
bash scripts/demo.sh log           # live: HEARD / IGNORED / SAID lines
bash scripts/k1voice.sh calibrate a1   # mic threshold for a noisy room (or the console's "calibrate mic" button)
```
YOLO detections as JSON: http://192.168.50.82:8092/detections, with boxes: http://192.168.50.82:8092/vis.jpg

## Troubleshooting

| Symptom | Fix |
|---|---|
| Ignores everything | Say the name first. The console's yellow line shows the wake words. `demo.sh log` shows what ASR heard: if the name comes out misspelled, add the spelling to `K1_WAKE_WORDS` in `scripts/k1voice.sh` (a1 line). |
| Hears nothing at all | Status shows laptop ASR DOWN: `demo.sh up` restarts it. Noisy room: calibrate mic. |
| Says it moved but did not / "not in walking mode" | Console: walking mode, or "Apollo, switch to walking mode". Check that STOP is not latched (press chat motion ON). |
| "perception unavailable", will not walk forward | YOLO service down: `bash scripts/k1voice.sh detect a1 start` |
| Long silence, then no reply | Free OpenRouter model is rate-limited or slow. Console System panel → chat LLM. Wait or retry. |
| Replies show in the console but are not spoken | Robot memory: `demo.sh status` (RAM under 500 MB is bad). `demo.sh down && demo.sh up`. |
| Robot has no internet (LLM errors) | Laptop rebooted, so NAT was lost: `demo.sh up` re-creates it. Robot DNS must be 8.8.8.8 (docs: network notes in booster_ws). |
| Robot fell / rebooted | Charge it, stand it up with the remote, then `demo.sh up`. |

## Settings (set on the laptop before `demo.sh up`)

| Var | Default | Meaning |
|---|---|---|
| `K1_CALL_NAME` / `K1_WAKE_WORDS` | Apollo / apollo variants (a1 only) | name and wake phrases; `K1_WAKE_WORDS=off` answers everything |
| `K1_WAKE_WINDOW_S` | 0 | seconds of free follow-ups after each reply (0 = name every time) |
| `K1_KOKORO_VOICE` / `K1_KOKORO_LANG` | `bf_isabella` / `en-gb` | voice |
| `K1_DETECT` | on | YOLO/depth client in the agent |
| `.env`: `K1_CHAT_MODEL`, `K1_VISION_MODEL`, `OPENROUTER_API_KEY` | n/a | free models only (`:free`); never commit `.env` |
