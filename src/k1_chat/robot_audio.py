"""On-robot audio: arecord -> Segmenter -> faster-whisper, Kokoro (GPU) -> aplay. Run via robot/run.sh (needs the mic free).

    python -m k1_chat.robot_audio smoke "Hello, I am a K1."   # speak, hear one utterance, say it back
    python -m k1_chat.robot_audio listen                      # print every utterance until killed
"""
import argparse
import io
import os
import select
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from k1_chat import debug
from k1_chat.audio import Audio, LiveCalibration, Segmenter, vad_threshold
from k1_chat.conversation import SYSTEM, WAKE_WORDS, WakeGate, is_stop

RATE, FRAME_MS = 16000, 30
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2
MIC = "plughw:CARD=Speaker,DEV=0"
SPEAKER = "plughw:CARD=Device,DEV=0"
HOME = Path(os.environ.get("K1_CHAT_HOME", Path.home() / "k1_chat"))


LIVE_CH, NCH = 4, 6  # the K1 mic array exposes 6 channels; 0-3 are live mics, 4-5 are silent (rms ~3)


def set_mic_gain(level: int | None = None) -> None:
    """Capture volume is 10/25 by default (peaks ~1500/32768); ALSA mixer setting, lost on reboot."""
    level = int(os.environ.get("K1_CHAT_MIC_GAIN", 16)) if level is None else level
    subprocess.run(["amixer", "-q", "-c", "Speaker", "cset", "name=Mic Capture Volume", ",".join([str(level)] * NCH)], check=False)


class ArecordCapture:
    """6-channel capture mixed down to mono (mean of the live channels); ALSA's own 6->1 mix includes the dead ones and loses ~40%."""

    def __init__(self, device: str = MIC):
        set_mic_gain()
        self._p = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(RATE), "-c", str(NCH), "-t", "raw"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def __iter__(self):
        import numpy as np
        size = FRAME_BYTES * NCH
        while chunk := self._p.stdout.read(size):
            if len(chunk) == size:
                x = np.frombuffer(chunk, np.int16).reshape(-1, NCH)[:, :LIVE_CH]
                yield x.astype(np.int32).mean(axis=1).astype(np.int16).tobytes()

    def drain(self) -> None:
        fd = self._p.stdout.fileno()
        while select.select([fd], [], [], 0)[0]:
            os.read(fd, 65536)

    def close(self) -> None:
        self._p.terminate()


class WhisperAsr:
    def __init__(self, model: str | None = None):
        from faster_whisper import WhisperModel
        model = model or os.environ.get("K1_WHISPER_MODEL", "tiny.en")  # CPU int8 on the K1 (no CUDA ctranslate2)
        self._m = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=4)

    def __call__(self, pcm: bytes) -> str:
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = self._m.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        return " ".join(s.text for s in segs)


class NemotronAsr:
    """Nemotron-Labs-Unified-ASR (NeMo fp16) via the local nemo-asrd daemon,127.0.0.1:8090.

    Same call shape as WhisperAsr (S16LE mono 16 kHz PCM -> text). The daemon keeps the model
    resident (28 s cold load); see docs/PLAN_VOICE_VLM_2026-10-28.md §9.6 and
    PLAN_WAKEWORD_NEMO.md §3.2 N3. On any daemon failure returns "" and logs —
    select the backend with K1_CHAT_ASR=nemo|whisper (see make_asr).
    """

    MIN_BYTES = 1600 * 2  # <100 ms of audio: nothing to transcribe, skip the HTTP round trip
    URL = "http://127.0.0.1:8090/transcribe"

    def __init__(self, url: str | None = None, lang: str = "en", timeout: float = 25.0):
        self._url = url or os.environ.get("K1_NEMOTRON_URL", self.URL)
        self._lang = lang or os.environ.get("K1_NEMOTRON_LANG", "en")
        self._timeout = timeout

    def __call__(self, pcm: bytes) -> str:
        if len(pcm) < self.MIN_BYTES:
            return ""
        import base64
        import json
        import urllib.request

        payload = json.dumps({"pcm16": base64.b64encode(pcm).decode(), "sr": RATE, "lang": self._lang}).encode()
        req = urllib.request.Request(self._url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                out = json.loads(r.read())
            return str(out.get("text", ""))
        except Exception as e:  # noqa: BLE001 — ASR failure must not kill the audio loop
            print(f"NemotronAsr: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return ""

    def health(self) -> bool:
        import urllib.request

        try:
            with urllib.request.urlopen(self._url.rsplit("/", 1)[0] + "/health", timeout=2.0) as r:
                return json_ok(r.read())
        except Exception:  # noqa: BLE001
            return False


def json_ok(body: bytes) -> bool:
    import json

    try:
        return bool(json.loads(body).get("ready"))
    except Exception:  # noqa: BLE001
        return False


def make_asr():
    """Pick the ASR backend. K1_CHAT_ASR=nemo (default) -> local NeMo daemon, but falls back to
    WhisperAsr when nemo-asrd is not running (other robots, daemon crash) so chat stays alive."""
    backend = os.environ.get("K1_CHAT_ASR", "nemo").lower()
    if backend in ("nemo", "nemotron"):
        asr = NemotronAsr()
        if asr.health():
            return asr
        print("K1_CHAT_ASR=nemo but nemo-asrd is not healthy -> WhisperAsr fallback", file=sys.stderr, flush=True)
    elif backend not in ("whisper", ""):
        print(f"unknown K1_CHAT_ASR={backend!r} -> WhisperAsr", file=sys.stderr, flush=True)
    return WhisperAsr()


def clauses(text: str, limit: int = 120) -> list[str]:
    """Split text longer than limit at commas/dashes/semicolons into pieces <= limit where possible (Kokoro memory)."""
    import re
    out = []
    for piece in re.split(r"(?<=[,;:\u2014])\s+|\s+[\u2014-]\s+", text):
        if out and len(out[-1]) + len(piece) + 1 <= limit:
            out[-1] += " " + piece
        else:
            out.append(piece)
    return [p for p in out if p.strip()] or [text]


class KokoroTts:
    """Kokoro-82M via the system onnxruntime-gpu (CUDA). ~0.2x real time on the K1's Orin; CPU is ~2.8x."""

    def __init__(self, model: Path = HOME / "models" / "kokoro-v1.0.onnx", voices: Path = HOME / "models" / "voices-v1.0.bin",
                 voice: str | None = None, lang: str | None = None, speed: float | None = None):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        device = os.environ.get("K1_KOKORO_DEVICE", "gpu").lower()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            # Cap the CUDA arena: on the Orin's unified memory an uncapped arena is system RAM,
            # and the default (multi-GiB) starves nemo-asrd -> the ASR daemon gets OOM-killed
            # mid-transcribe (PLAN_VOICE_VLM §9.7). 512 MB fits the 82M model + activations.
            limit = int(os.environ.get("K1_KOKORO_CUDA_MEM_LIMIT", str(512 * 1024 * 1024)))
            providers = [
                ("CUDAExecutionProvider", {"gpu_mem_limit": limit, "arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider",
            ]
        sess = ort.InferenceSession(str(model), providers=providers)
        if sess.get_providers()[0] != "CUDAExecutionProvider":
            print("WARNING: Kokoro is on CPU, expect ~3x slower than real time", file=sys.stderr, flush=True)
        self._k = Kokoro.from_session(sess, str(voices))
        # K1_KOKORO_VOICE: a voice name or a blend "bf_emma:0.6+bf_isabella:0.4" (default: British female)
        self._voice = voice or os.environ.get("K1_KOKORO_VOICE", "bf_isabella")
        self._lang = lang or os.environ.get("K1_KOKORO_LANG", "en-gb")
        self._speed = speed or float(os.environ.get("K1_KOKORO_SPEED", "1.0"))

    def _style(self, spec: str):
        if "+" not in spec and ":" not in spec:
            return spec
        parts = [p.split(":") for p in spec.split("+")]
        return sum(float(w) * self._k.get_voice_style(n) for n, w in parts)

    def _synth(self, text: str):
        """Long text needs a bigger CUDA alloc than the capped arena has left (BFCArena 'Available memory of ...'):
        on failure split in half and retry, so one long sentence never kills the speak thread."""
        import numpy as np
        try:
            return self._k.create(text, voice=self._style(self._voice), speed=self._speed, lang=self._lang)
        except Exception as e:  # noqa: BLE001 — onnxruntime RuntimeException
            words = text.split()
            if len(words) < 4:
                print(f"[tts] skipped {text!r}: {str(e)[:120]}", file=sys.stderr, flush=True)
                return np.zeros(0, np.float32), 24000
            a, sr = self._synth(" ".join(words[:len(words) // 2]))
            b, sr = self._synth(" ".join(words[len(words) // 2:]))
            return np.concatenate([a, b]), sr

    def __call__(self, text: str) -> bytes:
        import numpy as np
        parts = clauses(text)
        out = [self._synth(t) for t in parts]
        samples, sr = np.concatenate([o[0] for o in out]), out[0][1]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue()


class AplayPlayer:
    def __init__(self, device: str = SPEAKER):
        self._dev, self._p = device, None

    def play(self, wav: bytes) -> None:
        self._p = subprocess.Popen(["aplay", "-q", "-D", self._dev, "-"], stdin=subprocess.PIPE)
        try:
            self._p.communicate(wav)
        except BrokenPipeError:
            pass

    def stop(self) -> None:
        if self._p and self._p.poll() is None:
            self._p.terminate()


DUMP = HOME / "dump"


def save_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)


def _dumped(asr, seg):
    """Wrap ASR: keep the exact PCM the model receives (utt_*.wav) plus a .txt with level stats, threshold and the text."""
    import numpy as np

    def wrapped(pcm: bytes) -> str:
        text = asr(pcm)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        x = np.frombuffer(pcm, np.int16).astype(np.float64)
        save_wav(DUMP / f"utt_{stamp}.wav", pcm)
        (DUMP / f"utt_{stamp}.txt").write_text(
            f"text: {text!r}\nsec: {len(x) / RATE:.2f}\nrms: {np.sqrt(np.mean(x ** 2)):.0f}\npeak: {int(np.abs(x).max())}\n"
            f"start_rms_threshold: {seg._start_rms:.0f}\n")
        for old in sorted(DUMP.glob("utt_*.wav"))[:-30]:  # keep the last 30
            old.unlink(); old.with_suffix(".txt").unlink(missing_ok=True)
        return text
    return wrapped


def _timed(name, fn, size=len):
    def wrapped(x):
        t = time.time()
        out = fn(x)
        print(f"[{name}] {time.time() - t:.2f}s in={size(x)} out={out if isinstance(out, str) else len(out)}", file=sys.stderr, flush=True)
        debug.note(name, time.time() - t, detail=f"in={size(x)} out={out if isinstance(out, str) else len(out)}")
        return out
    return wrapped


def build(concurrent: bool = False) -> Audio:
    asr = _timed("asr", make_asr(), size=lambda pcm: f"{len(pcm) / 2 / RATE:.1f}s audio")
    tts = _timed("tts", KokoroTts(), size=lambda t: f"{len(t)} chars")
    start_rms = float(os.environ.get("K1_CHAT_START_RMS", _saved_rms()))
    seg = Segmenter(RATE, FRAME_MS, start_rms=start_rms, end_silence_ms=450)
    return Audio(ArecordCapture(), _dumped(asr, seg), tts, AplayPlayer(), seg, concurrent=concurrent)


def _saved_rms(default: float = 150.0) -> float:
    try:
        return float((HOME / "start_rms").read_text())
    except (OSError, ValueError):
        return default


def mic_test(secs: int = 6) -> int:
    """Record all 6 mic-array channels while you speak; print level + ASR per channel and for the mono mix, keep wavs."""
    import numpy as np
    set_mic_gain()
    print(f"SPEAK NOW for {secs}s (a full sentence, at the distance you will use) ...", flush=True)
    raw = subprocess.run(["arecord", "-q", "-D", MIC, "-f", "S16_LE", "-r", str(RATE), "-c", "6", "-t", "raw", "-d", str(secs)],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw[: len(raw) // 12 * 12], np.int16).reshape(-1, 6)
    asr = make_asr()
    clips = {f"ch{i}": x[:, i] for i in range(6)}
    clips["mean"] = x.astype(np.float32).mean(axis=1).astype(np.int16)
    for name, c in clips.items():
        c = np.ascontiguousarray(c, np.int16)
        save_wav(DUMP / f"mictest_{name}.wav", c.tobytes())
        rms = float(np.sqrt(np.mean(c.astype(np.float64) ** 2))); peak = int(np.abs(c).max())
        print(f"{name:5s} rms {rms:6.0f} peak {peak:6d}  ASR: {asr(c.tobytes())!r}", flush=True)
    return 0


def calibrate(quiet_s: int = 3, speak_s: int = 5) -> int:
    """Measure room noise, then speech, save a VAD threshold to ~/k1_chat/start_rms and transcribe the speech clip."""
    cap = ArecordCapture()

    def record(secs):
        frames = []
        for f in cap:
            frames.append(f)
            if len(frames) * FRAME_MS >= secs * 1000:
                return frames
        return frames

    def rms(f):
        return Segmenter._rms(f)

    print(f"Stay SILENT for {quiet_s}s ...", flush=True)
    cap.drain()
    quiet_frames = record(quiet_s)
    noise = sorted(rms(f) for f in quiet_frames)
    print(f"SPEAK NOW for {speak_s}s (say a sentence at normal volume, standing where you will talk) ...", flush=True)
    cap.drain()
    speech_frames = record(speak_s)
    cap.close()
    save_wav(DUMP / "calib_quiet.wav", b"".join(quiet_frames))
    save_wav(DUMP / "calib_speech.wav", b"".join(speech_frames))
    r = vad_threshold(noise, [rms(f) for f in speech_frames])
    n95, s_med, s_top, thr = r["noise_p95"], r["speech_median"], r["speech_peak"], float(r["threshold"])
    print(f"noise p95 rms {n95:.0f}   speech median-of-loud-half {s_med:.0f}   speech peak {s_top:.0f}")
    if not r["ok"]:
        print("WARN: speech barely above noise: speak louder/closer, or the mic is quiet. Threshold NOT saved.")
        verdict = 1
    else:
        HOME.mkdir(parents=True, exist_ok=True)
        (HOME / "start_rms").write_text(f"{thr:.0f}")
        print(f"saved start_rms = {thr:.0f}")
        verdict = 0
    heard = make_asr()(b"".join(speech_frames))
    (DUMP / "calib_report.txt").write_text(
        f"noise_p95_rms: {n95:.0f}\nspeech_median_loud_half_rms: {s_med:.0f}\nspeech_peak_rms: {s_top:.0f}\n"
        f"threshold_saved: {f'{thr:.0f}' if verdict == 0 else 'no'}\nasr_text: {heard!r}\n")
    print("ASR heard:", repr(heard), flush=True)
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("sentence", nargs="?", default="Hello, I am a Booster K1. Please say something.")
    sm.add_argument("--timeout", type=int, default=20)
    sub.add_parser("listen")
    sub.add_parser("chat")  # needs OPENROUTER_API_KEY + K1_CHAT_MODEL in the environment
    sub.add_parser("agent")  # chat + robot tools + console :8091; K1_ROBOT_NAME, K1_BACKEND=ros|sim, K1_VISION_MODEL optional
    sub.add_parser("bench")
    sub.add_parser("calibrate")
    sub.add_parser("mic-test")
    af = sub.add_parser("asr-file")
    af.add_argument("wav")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.cmd == "calibrate":
        return calibrate()
    if args.cmd == "mic-test":
        return mic_test()
    a = build(concurrent=args.cmd in ("chat", "agent"))
    try:
        if args.cmd == "bench":
            wav = a._tts("Hello, I am a Booster K1 robot. It is nice to meet you. How can I help you today?")
            for _ in range(2):
                a._tts("Hello, I am a Booster K1 robot. It is nice to meet you. How can I help you today?")
            import numpy as np
            with wave.open(io.BytesIO(wav)) as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
                n = int(len(x) * RATE / w.getframerate())
                pcm16 = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16).tobytes()
            print("ASR:", a._asr(pcm16), flush=True)
            print("ASR:", a._asr(pcm16), flush=True)
        elif args.cmd == "asr-file":
            with wave.open(args.wav) as w:
                assert w.getframerate() == RATE and w.getnchannels() == 1, "need 16 kHz mono wav"
                print("ASR:", a._asr(w.readframes(w.getnframes())), flush=True)
        elif args.cmd == "smoke":
            a.speak(args.sentence)
            print("listening...", flush=True)
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
            signal.alarm(args.timeout)
            try:
                heard = next(a.listen())
            except (TimeoutError, StopIteration):
                print(f"HEARD: <nothing> (mic peak rms {a._seg.peak:.0f}, threshold {a._seg._start_rms:.0f})", flush=True)
                return 1
            finally:
                signal.alarm(0)
            print(f"HEARD: {heard}", flush=True)
            a.speak(f"I heard: {heard}")
        elif args.cmd == "chat":
            from k1_chat.conversation import Conversation, run_async_voice_chat
            from k1_chat.llm import from_env

            def heard():
                for text in a.listen():
                    print(f"HEARD: {text}", flush=True)
                    yield text

            def say(text):
                t = time.time()
                print(f"SAID: {text}", flush=True)
                a.speak(text)
                print(f"[speak] {time.time() - t:.2f}s", file=sys.stderr, flush=True)

            run_async_voice_chat(heard(), Conversation(from_env(), history_file=HOME / "history.json"), say)
        elif args.cmd == "agent":
            from k1_chat.brain import Brain, make_backend
            from k1_chat.llm import from_env

            battery = None  # robot_status tool; None -> "battery unknown"
            if os.environ.get("K1_BACKEND", "ros") == "ros":
                try:
                    from k1_chat.booster_robot import RosBattery
                    battery = RosBattery().read
                except Exception as e:
                    print(f"[agent] no battery reading: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            frames = None  # head camera for the look/watch tools and the console's live view
            if os.environ.get("K1_VISION_MODEL"):
                if os.environ.get("K1_BACKEND", "ros") == "ros":
                    from k1_chat.booster_robot import RosFrames
                    frames = RosFrames()
                else:
                    try:
                        from k1_chat.frames import StudioFrames
                        frames = StudioFrames("127.0.0.1")
                    except ImportError:
                        pass

            def look_tool(reg, log):  # feat/k1-vision; frames from ROS when it is up, else the Studio websocket
                try:
                    from k1_chat.vision import add_look_tool
                except ImportError:
                    print("[agent] K1_VISION_MODEL set but vision module missing; no look tool", file=sys.stderr, flush=True)
                    return
                if frames:
                    add_look_tool(reg, frames, from_env("K1_VISION_MODEL"), log)

            perception = None  # detect_server (k1voice.sh detect <robot> start): obstacle guard, detect + follow tools
            if os.environ.get("K1_DETECT", "on") != "off":
                from k1_chat.perception import Perception
                perception = Perception()
                print(f"[agent] perception: {'up' if perception.get() else 'DOWN (forward walking refused until it is up)'}",
                      flush=True)
            call = os.environ.get("K1_CALL_NAME")  # what people call this robot, e.g. Apollo for A1
            system = SYSTEM + (f"\n\nYour name is {call}. People address you as {call}; you only hear what was said "
                               f"to you by name, other room talk is not for you." if call else "")
            b = Brain(os.environ.get("K1_ROBOT_NAME", "K1"), make_backend(os.environ.get("K1_BACKEND", "ros")),
                      from_env(), system=system, history_file=HOME / "history.json",
                      extra_tools=look_tool if os.environ.get("K1_VISION_MODEL") else None, battery=battery,
                      perception=perception)
            cal = LiveCalibration(a._seg, FRAME_MS, lambda thr: (HOME / "start_rms").write_text(f"{thr:.0f}"))
            a.tap = cal.feed  # console "calibrate mic" button; the agent keeps the mic
            live = frames.grab if frames else None
            if perception and frames:
                live = lambda: perception.vis() or frames.grab()  # noqa: E731 — YOLO boxes when detect is up
            words = os.environ.get("K1_WAKE_WORDS") or WAKE_WORDS
            window = float(os.environ.get("K1_WAKE_WINDOW_S") or 0)
            first = words.split(",")[0]
            wake = "wake words off: robot answers everything" if words == "off" else (
                f"Wake words: {words.replace(',', ' / ')}  (every command, e.g. \"{first}, turn left\"; "
                f"\"{first}\" alone -> \"Yes?\" then one command; "
                + (f"follow-ups free for {window:.0f} s after it speaks; " if window else "")
                + "\"stop\" always works)")
            url = b.serve_console("0.0.0.0", 8091, live=live, debug=debug.collect, calibrate=cal, wake=wake)
            print(f"console: {url}", flush=True)
            speak_lock = threading.Lock()
            gate = None if words == "off" else WakeGate(words, window)
            print(f"[agent] wake words: {words}", flush=True)

            def handle(text):
                print(f"HEARD: {text}", flush=True)
                answer = b.handle(text) if text else "Yes?"
                if answer is not None:
                    with speak_lock:
                        print(f"SAID: {answer}", flush=True)
                        a.speak(answer)
                    if gate:
                        gate.touch()  # K1_WAKE_WINDOW_S > 0: follow-ups need no wake word for a while
                        if b.listen_next or answer.rstrip().endswith("?"):
                            gate.arm()  # keep_listening or a question: the next sentence needs no name
                            print("[agent] listening for one reply without wake word", flush=True)

            for text in a.listen():
                if is_stop(text):
                    handle(text)  # e-stop inline, never queued behind a thinking LLM, no wake word needed
                    continue
                if gate:
                    passed = gate.check(text)
                    if passed is None:
                        b.log.add("ignored", text=text)
                        print(f"IGNORED (no wake word): {text}", flush=True)
                        continue
                    text = passed
                threading.Thread(target=handle, args=(text,), daemon=True).start()
        else:
            for text in a.listen():
                print(f"HEARD: {text}", flush=True)
    finally:
        a._cap.close()
        a.stop_speaking()
    return 0


if __name__ == "__main__":
    sys.exit(main())
