# Benchmarking

smoltalk ships the instrumentation and harness used to measure its changes. For reference: the warm whisper-rs provider changed measured stop-to-text from 3,163 ms p50 / 5,300 ms p95 (per-utterance `whisper-cli` subprocess) to 1,108 ms p50 / 1,280 ms p95 on real speech.

## Trace sink

The daemon emits JSONL trace events when `CHEZWIZPER_BENCH_TRACE` is set:

```bash
CHEZWIZPER_BENCH_TRACE=/tmp/events.jsonl chezwizper
```

Every phase of the dictation path is stamped with monotonic nanosecond timings: API receive, main-loop dequeue, audio start / first sample / stop, samples taken, WAV write, transcription begin/end, provider spawn/exit, clipboard copy, injection begin/end, and idle/error transitions. When the variable is unset the sink is a no-op — zero overhead in normal use.

Experimental pause chunking also emits `chunk_transcription_begin` and `chunk_transcription_end`, including session ID, chunk index, sample count, and success.

Optional metadata env vars: `CHEZWIZPER_BENCH_RUN_ID`, `CHEZWIZPER_BENCH_TRIAL_ID`, `CHEZWIZPER_BENCH_PHRASE_ID`.

## Harness

The `bench/` directory contains the end-to-end harness:

| Script | Purpose |
|--------|---------|
| `make_corpus.py` | Materialize the 30-phrase corpus as WAVs (TTS via `flite`, silence fallback with a `tts` tag) |
| `bench_e2e.py` | Drive a scratch daemon end-to-end and derive per-trial phase timings |
| `bench_transcribe.py` | Offline provider benchmark over the corpus WAVs with WER/CER |
| `summarize.py` | Aggregate trials into `summary.csv` and `baseline_report.md` |
| `test_bench_harness.py` | Regression tests for the harness itself |

### Running an end-to-end benchmark

```bash
cargo build --release
python3 bench/make_corpus.py --tts flite --output-dir bench-artifacts/corpus
python3 bench/bench_e2e.py --output-dir bench-artifacts/my-run --provider whisper-rs --model base.en
```

The harness is designed to run **alongside your live service**:

- It derives its own daemon config from `--base-config` (default `~/.config/chezwizper/config.toml`) with safe overrides — port **3838** (via `[api] port`), no auto-paste, no notifications, no sounds — and launches a scratch daemon from `target/release/`.
- Audio goes through a PipeWire loopback: a transient null sink is created with `pactl`, the scratch daemon's input is redirected with `PULSE_SOURCE=<sink>.monitor`, and each corpus phrase is played into the sink with `paplay`. Your microphone and speakers are untouched, and the sink is unloaded on exit.
- Per-trial RSS/CPU is sampled from `/proc` across the daemon's process tree.

### Outputs

Each run directory contains:

- `events.jsonl` — every trace event, with `bench_trial_begin`/`bench_trial_end` markers delimiting trials
- `trials.jsonl` — one row per trial: phase timings, RTF, peak RSS, CPU, status, and `total_stop_to_text_source` (`injection`, `clipboard`, or `idle` — states honestly which endpoint the total was measured to)
- `summary.csv` — percentiles per phrase group
- `baseline_report.md` — human-readable summary

### Interpreting results

- `wav_to_transcription_ms` dominates everything else in the pipeline (capture, WAV write, and clipboard are ~60 ms combined).
- Whisper pads every clip to a fixed 30-second encoder window, so transcription time is roughly **constant with respect to clip length** unless `audio_ctx` shrinking is enabled (`audio_ctx = "auto"` in `[whisper]`).
- Injection timing is only measured with `--measure-injection`, which enables auto-paste in the scratch daemon — text will be typed into the focused window, so give it a sacrificial editor window.

### Known limitations

- WER/CER is computed by `bench_transcribe.py` (offline) only; the e2e runner does not yet capture transcript text.
- `flite` TTS is robotic — WER numbers are a relative signal between configurations, not an absolute accuracy claim.
