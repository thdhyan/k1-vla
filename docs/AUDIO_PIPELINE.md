# Audio pipeline (robot side, fw 1.8.x)

mic (USB array, `plughw:CARD=Speaker`, 6 ch captured, mean of live ch 0-3 -> mono; mic gain set to 16/25, `K1_CHAT_MIC_GAIN`) -> 16 kHz S16, 30 ms frames
-> `Segmenter`: frame rms >= `start_rms` starts an utterance (300 ms pre-roll kept), 450 ms of quiet ends it
-> raw PCM of the utterance (no filtering/AGC/normalisation) -> faster-whisper `tiny.en` (CPU int8, beam 1, no VAD) -> text.

`start_rms`: env `K1_CHAT_START_RMS`, else `~/k1_chat/start_rms` (written by `calibrate`), else 150.
`calibrate`: 3 s silence -> noise p95 rms; 5 s speech -> median of the loud half + peak;
threshold = max(60, 2.5 x noise p95), or the midpoint of noise p95 and speech median if that is too high. Not saved if speech peak < 2 x noise p95.

## Dumps (`~/k1_chat/dump/`, `k1voice.sh fetch <robot>` -> `audio_dumps/<robot>/`)
- `calib_quiet.wav`, `calib_speech.wav`, `calib_report.txt`: exactly what calibration measured, and its numbers.
- `mictest_ch0..5.wav`, `mictest_mean.wav`: each array channel and the mean, from `mic-test`.
- `utt_<time>.wav` + `.txt`: the exact PCM given to Whisper for every utterance (last 30), with text, rms, peak and the threshold used.
