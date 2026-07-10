# Pause-Triggered Chunking Experiment

smoltalk includes an opt-in experiment that transcribes completed speech segments during natural pauses and injects the stitched transcript once on push-to-talk release. It is designed to reduce release-to-text latency for long dictation turns, not to provide live partial text.

## Design

- Poll the active 16 kHz capture buffer without interrupting recording.
- Detect speech with the configured amplitude threshold.
- Flush only after at least `min_chunk_ms` of speech and `pause_ms` of silence.
- Run one Whisper inference at a time on Tokio's blocking pool.
- Retain bounded audio overlap and remove only multi-word transcript overlap.
- At release, use the authoritative final VAD boundary to detect any uncovered tail.
- Fall back to the existing full-turn transcription if a chunk errors or produces empty text.
- Never inject partial text; injection still happens once after stitching.

## Initial A/B

The first experiment used one synthetic 25-second turn built from three real-speech corpus clips separated by 1.2-second pauses. The provider was warm `whisper-rs` with `base.en-q5_0`, `audio_ctx = "auto"`, no notifications, no sounds, and no injection measurement.

| Threads | Mode | Release-to-idle | CPU user time | Final temperature | Result |
|---:|---|---:|---:|---:|---|
| 4 | Final-only | 4,761 ms | 14,290 ms | 93 C | Baseline |
| 4 | Pause chunking | 7,970 ms | 39,480 ms | 92 C | 67% slower; sustained inference throttled |
| 2 | Final-only | 3,916 ms | 7,470 ms | 80 C | Baseline |
| 2 | Pause chunking | 1,803 ms | 8,450 ms | 77 C | 54% faster with 13% more CPU time |

With two threads, the first two chunks completed while recording continued. Release paid approximately 188 ms for stop/VAD/WAV work and 1,614 ms for the final uncovered chunk. With four threads, background inference exhausted the laptop's thermal budget and made the final chunk much slower.

These are single-trial directional results, not stable percentiles. The live ChezWizper service was active during this testing and may have been used concurrently; the scratch daemon used a separate API port and PipeWire loopback, so audio routing remained isolated, but CPU load and temperature may have been contaminated. Treat the numbers as evidence that the implementation overlaps work, not as an adoption benchmark.

The current end-to-end harness also does not capture transcript text, so no WER/CER or boundary duplicate/drop claim can be made from this run. Observed final character counts differed slightly between final-only and chunked output, reinforcing the need for an accuracy run before enabling chunking by default.

## Recommendation

Keep pause chunking disabled by default while collecting repeated latency and accuracy results. On the tested two-core/four-thread Intel CPU, configure Whisper with two threads when evaluating chunking:

```toml
[whisper]
threads = 2

[chunking]
enabled = true
pause_ms = 600
min_chunk_ms = 5000
overlap_ms = 300
```

The next decision gate is a temperature-controlled corpus with captured final text, WER/CER, duplicate/drop counts, and repeated trials across short turns, long turns without pauses, and long turns with natural pauses.
