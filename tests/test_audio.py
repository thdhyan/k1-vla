import struct

import pytest
from k1_chat.audio import Audio, LiveCalibration, Segmenter, vad_threshold

FRAME_MS = 30
N = 480  # samples per 30 ms @ 16 kHz


def frame(amp: int) -> bytes:
    return struct.pack(f"<{N}h", *([amp, -amp] * (N // 2)))


LOUD, QUIET = frame(3000), frame(10)


def seg(**kw):
    return Segmenter(rate=16000, frame_ms=FRAME_MS, start_rms=500, end_silence_ms=300, min_ms=200,
                     max_ms=2000, preroll_ms=60, **kw)


def run(s, frames):
    return [u for f in frames if (u := s.feed(f))]


def test_utterance_emitted_after_trailing_silence():
    out = run(seg(), [QUIET] * 5 + [LOUD] * 10 + [QUIET] * 12)
    assert len(out) == 1
    assert len(out[0]) >= 10 * len(LOUD)


def test_preroll_included():
    out = run(seg(), [QUIET] * 5 + [LOUD] * 10 + [QUIET] * 12)
    assert len(out[0]) >= (10 + 2) * len(LOUD)  # 60 ms preroll = 2 frames


def test_short_blip_ignored():
    assert run(seg(), [QUIET] * 3 + [LOUD] * 2 + [QUIET] * 15) == []


def test_silence_only_emits_nothing():
    assert run(seg(), [QUIET] * 50) == []


def test_max_length_forces_cut():
    out = run(seg(), [LOUD] * 100)
    assert len(out) >= 1
    assert all(len(u) <= 2000 // FRAME_MS * len(LOUD) + 3 * len(LOUD) for u in out)


def test_two_utterances():
    out = run(seg(), ([LOUD] * 10 + [QUIET] * 12) * 2)
    assert len(out) == 2


class Cap:
    def __init__(self, frames): self.frames, self.drained = frames, 0
    def __iter__(self): return iter(self.frames)
    def drain(self): self.drained += 1


def make(frames, asr=lambda pcm: "hello robot", tts=lambda t: b"WAV:" + t.encode()):
    played = []

    class Player:
        stopped = 0
        def play(self, wav): played.append(wav)
        def stop(self): self.stopped += 1

    p = Player()
    return Audio(Cap(frames), asr, tts, p, seg()), played, p


def test_listen_yields_transcripts():
    a, _, _ = make([LOUD] * 10 + [QUIET] * 12)
    assert list(a.listen()) == ["hello robot"]


@pytest.mark.parametrize("junk", ["", "  ", "you", "Thank you.", "[BLANK_AUDIO]"])
def test_listen_drops_empty_and_whisper_hallucinations(junk):
    a, _, _ = make([LOUD] * 10 + [QUIET] * 12, asr=lambda pcm: junk)
    assert list(a.listen()) == []


def test_speak_synthesizes_plays_and_drains_mic():
    a, played, _ = make([])
    a.speak("hi there")
    assert played == [b"WAV:hi there"]
    assert a._cap.drained == 1


def test_speak_plays_sentence_by_sentence_so_first_audio_starts_early():
    a, played, _ = make([])
    a.speak("Hello there. How are you today? Fine!")
    assert played == [b"WAV:Hello there.", b"WAV:How are you today?", b"WAV:Fine!"]
    assert a._cap.drained == 1


def test_speak_keeps_decimals_and_abbreviation_free_text_together():
    a, played, _ = make([])
    a.speak("Speed is 0.4 m/s now")
    assert played == [b"WAV:Speed is 0.4 m/s now"]


def test_speak_blank_is_noop():
    a, played, _ = make([])
    a.speak("  ")
    assert played == []


def test_stop_speaking_stops_player():
    a, _, p = make([])
    a.stop_speaking()
    assert p.stopped == 1


@pytest.mark.robot
def test_robot_speaks_and_hears():
    pytest.skip("run `scripts/k1voice.sh smoke <robot>` instead; needs a person at the robot")


def test_concurrent_mode_ignores_mic_while_speaking_and_skips_drain():
    frames = [LOUD] * 10 + [QUIET] * 12
    a, played, _ = make(frames)
    a._concurrent = True
    a._speaking = True                    # robot is talking: its own voice must not become an utterance
    assert list(a.listen()) == []
    a._speaking = False
    a._cap.frames = frames
    assert list(a.listen()) == ["hello robot"]
    a.speak("Hi.")
    assert a._cap.drained == 0            # listener thread owns the pipe, drain would split frames


def test_vad_threshold_matches_calibrate_rules():
    r = vad_threshold([10.0] * 100, [10.0] * 50 + [1000.0] * 50)
    assert r["ok"] and r["noise_p95"] == 10 and r["threshold"] == 60  # floor 60
    assert not vad_threshold([100.0] * 100, [150.0] * 100)["ok"]  # speech < 2x noise


def test_live_calibration_taps_mic_and_sets_threshold():
    import array
    frame = lambda amp: array.array("h", [amp] * 480).tobytes()  # 30 ms at 16 kHz
    saved = []
    seg = Segmenter(16000, 30, start_rms=500.0)
    cal = LiveCalibration(seg, 30, saved.append, quiet_s=0.3, speak_s=0.3)
    assert not cal.feed(frame(5)) and cal.status()["phase"] == "idle"  # idle: frame goes to the segmenter
    cal.start()
    assert cal.start()["error"]  # one run at a time
    for _ in range(10):
        assert cal.feed(frame(20))
    assert cal.status()["phase"] == "speak" and cal.status()["prompt"].startswith("SPEAK")
    for _ in range(10):
        cal.feed(frame(2000))
    st = cal.status()
    assert st["phase"] == "done" and st["result"]["ok"] and st["result"]["previous"] == 500
    assert seg._start_rms == st["result"]["threshold"] == saved[0] == 60  # max(60, 2.5 * noise 20)
    assert not cal.feed(frame(5))


def test_audio_tap_consumes_frames_before_segmenter():
    import array
    loud = array.array("h", [3000] * 480).tobytes()
    quiet = array.array("h", [0] * 480).tobytes()
    frames = [loud] * 20 + [quiet] * 30
    a = Audio(iter(frames), lambda pcm: "hello", lambda t: b"", None, Segmenter(16000, 30, start_rms=500.0))
    a.tap = lambda f: True
    assert list(a.listen()) == []
