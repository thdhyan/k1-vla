"""Local voice pipeline: mic frames -> VAD segmenter -> ASR text, and text -> TTS wav -> player.

Capture, ASR, TTS and player are injected so the logic is testable; the real ones live in
robot_audio.py and run on the K1. tests/test_audio.py defines the contract.
"""
import array
import math
import re
import statistics
import threading
import time
from collections import deque
from typing import Callable, Iterable, Iterator, Protocol

# Whisper emits these on near-silence; they are never real user speech.
_HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "[blank_audio]", "(silence)"}


class Player(Protocol):
    def play(self, wav: bytes) -> None: ...
    def stop(self) -> None: ...


class Segmenter:
    """Energy VAD over fixed-size 16-bit mono PCM frames. feed() returns a finished utterance or None."""

    def __init__(self, rate=16000, frame_ms=30, start_rms=500.0, end_silence_ms=700, min_ms=300,
                 max_ms=15000, preroll_ms=300):
        ms = lambda v: max(1, v // frame_ms)
        self._start_rms = start_rms
        self._end, self._min, self._max = ms(end_silence_ms), ms(min_ms), ms(max_ms)
        self._pre = deque(maxlen=ms(preroll_ms))
        self.peak = 0.0  # loudest frame RMS seen, for tuning start_rms
        self.reset()

    def reset(self) -> None:
        self._pre.clear()
        self._buf: list[bytes] = []
        self._speech = 0
        self._quiet = 0

    @staticmethod
    def _rms(f: bytes) -> float:
        s = array.array("h", f)
        return math.sqrt(sum(x * x for x in s) / len(s)) if s else 0.0

    def feed(self, f: bytes) -> bytes | None:
        rms = self._rms(f)
        self.peak = max(self.peak, rms)
        loud = rms >= self._start_rms
        if not self._buf:
            if not loud:
                self._pre.append(f)
                return None
            self._buf = list(self._pre) + [f]
            self._pre.clear()
            self._speech, self._quiet = 1, 0
            return None
        self._buf.append(f)
        if loud:
            self._speech += 1
            self._quiet = 0
        else:
            self._quiet += 1
        if self._quiet >= self._end or len(self._buf) >= self._max:
            utt, ok = b"".join(self._buf), self._speech >= self._min
            self.reset()
            return utt if ok else None
        return None


class Audio:
    def __init__(self, capture: Iterable[bytes], asr: Callable[[bytes], str], tts: Callable[[str], bytes],
                 player: Player, segmenter: Segmenter, concurrent: bool = False, echo_tail_s: float = 0.5):
        self._cap, self._asr, self._tts, self._player, self._seg = capture, asr, tts, player, segmenter
        # concurrent: listen() runs in another thread while we speak; it alone reads the mic and discards our own voice
        self._concurrent, self._echo_tail, self._speaking, self._mute_until = concurrent, echo_tail_s, False, 0.0
        self.tap: Callable[[bytes], bool] | None = None  # e.g. LiveCalibration.feed; True = frame consumed

    def listen(self) -> Iterator[str]:
        for f in self._cap:
            if self._speaking or time.monotonic() < self._mute_until:
                self._seg.reset()
                continue
            if self.tap is not None and self.tap(f):
                self._seg.reset()
                continue
            utt = self._seg.feed(f)
            if utt is None:
                continue
            text = self._asr(utt).strip()
            if text.lower().strip(".!? ") not in _HALLUCINATIONS:
                yield text

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._speaking = True
        try:
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):  # first audio starts after one sentence, not all
                if sentence:
                    self._player.play(self._tts(sentence))
        finally:
            self._mute_until = time.monotonic() + self._echo_tail
            self._speaking = False
        if not self._concurrent:
            self._seg.reset()
            self._cap.drain()  # drop what the mic heard of our own voice

    def stop_speaking(self) -> None:
        self._player.stop()


def vad_threshold(noise: list[float], speech: list[float]) -> dict:
    """Start-RMS threshold from frame RMS of a quiet clip and a speech clip (k1voice calibrate + console button).
    ok=False when speech is barely above noise: keep the old threshold."""
    noise, loud = sorted(noise), sorted(speech)
    n95, s_med, s_top = noise[int(len(noise) * .95) - 1], statistics.median(loud[len(loud) // 2:]), loud[-1]
    thr = max(60.0, n95 * 2.5)
    if thr > s_med * 0.5:
        thr = max(60.0, (n95 + s_med) / 2) if s_med > n95 else thr
    return {"threshold": round(thr), "noise_p95": round(n95), "speech_median": round(s_med), "speech_peak": round(s_top),
            "ok": s_top >= n95 * 2}


class LiveCalibration:
    """Mic calibration while the agent owns the mic: set as Audio.tap, it eats `quiet_s` of silence then `speak_s`
    of speech, sets the segmenter threshold and calls save(threshold). status() is polled by the console."""

    def __init__(self, seg: Segmenter, frame_ms: int, save: Callable[[float], None], quiet_s: float = 3, speak_s: float = 5):
        self._seg, self._frame_ms, self._save, self._need = seg, frame_ms, save, {"quiet": quiet_s, "speak": speak_s}
        self._lock, self._phase, self._rms, self._result = threading.Lock(), "idle", {"quiet": [], "speak": []}, None

    def start(self) -> dict:
        with self._lock:
            if self._phase in ("quiet", "speak"):
                return {"error": "calibration already running", **self._status()}
            self._phase, self._rms, self._result = "quiet", {"quiet": [], "speak": []}, None
            return self._status()

    def feed(self, f: bytes) -> bool:
        with self._lock:
            if self._phase not in ("quiet", "speak"):
                return False
            buf = self._rms[self._phase]
            buf.append(Segmenter._rms(f))
            if len(buf) * self._frame_ms >= self._need[self._phase] * 1000:
                if self._phase == "quiet":
                    self._phase = "speak"
                else:
                    self._finish()
            return True

    def _finish(self) -> None:
        r = vad_threshold(self._rms["quiet"], self._rms["speak"])
        r["previous"] = round(self._seg._start_rms)
        if r["ok"]:
            self._seg._start_rms = float(r["threshold"])
            self._save(float(r["threshold"]))
        self._phase, self._result = "done", r

    def _status(self) -> dict:
        prompt = {"idle": "", "quiet": "stay SILENT", "speak": "SPEAK NOW: a normal sentence from where you will talk",
                  "done": "done"}[self._phase]
        left = 0.0
        if self._phase in ("quiet", "speak"):
            left = self._need[self._phase] - len(self._rms[self._phase]) * self._frame_ms / 1000
        return {"phase": self._phase, "prompt": prompt, "seconds_left": round(max(0.0, left), 1),
                "threshold": round(self._seg._start_rms), "result": self._result}

    def status(self) -> dict:
        with self._lock:
            return self._status()
