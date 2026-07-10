# Architecture

This document describes how smoltalk is put together: the modules, the data flow from keypress to injected text, and the design decisions behind the latency-sensitive paths.

> **Naming note:** the binary, crate paths, and trace/env names still use `chezwizper` while the rename is in progress.

## Overview

smoltalk is a single-process Tokio daemon. An HTTP API receives start/stop/toggle commands (typically bound to Hyprland keybinds via `curl`), audio is captured from the microphone, transcribed by a pluggable Whisper provider, normalized, and injected into the focused window.

```
 Hyprland keybind ──curl──▶ HTTP API (axum, 127.0.0.1:3737)
                                │  ApiCommand over mpsc
                                ▼
                         Main event loop (main.rs)
                                │
        ┌───────────┬───────────┼──────────────┬─────────────┐
        ▼           ▼           ▼              ▼             ▼
   AudioStream   VAD trim   Transcription   TextInjector   Indicator
   Manager       (Silero/   Service         (wtype/        (hyprctl
   (cpal, f32)   amplitude) (provider +     ydotool +      notify,
                            normalizer)     clipboard      sounds)
                                            paste)
```

Everything lives in one crate. `src/main.rs` wires the components together; `src/lib.rs` re-exports the modules for the `examples/`, `src/bin/test_api.rs`, and the test suite.

## Module map

| Module | Responsibility |
|--------|----------------|
| `api` | axum HTTP server, request→command reservation, app status state machine, Waybar status responses |
| `audio` | cpal input stream lifecycle, 16 kHz mono f32 sample buffer, VAD invocation on stop, and shared WAV encoding for file-based providers |
| `vad` | Voice activity detection: Silero (via whisper-rs) with amplitude-gate fallback; trims silence and can skip transcription entirely |
| `whisper` | `WhisperTranscriber` facade, `TranscriptionProvider` trait, provider selection/auto-detection |
| `whisper::providers` | The four providers: `whisper_rs` (in-process), `whisper_cpp` (CLI subprocess), `openai_cli`, `openai_api` |
| `transcription` | `TranscriptionService`: composes a `WhisperTranscriber` with the matching `Normalizer` |
| `normalizer` | Strips timestamps and non-speech markers from raw provider output |
| `chunking` | Pause-triggered chunked transcription: segments speech during recording and stitches overlapping transcripts |
| `text_injection` | Hybrid injection: direct typing (wtype/ydotool) or clipboard paste transaction with restore |
| `clipboard` | Standalone `ClipboardManager` helper (wl-copy with fallbacks); the injection path uses its own backend table |
| `ui` | `Indicator`: hyprctl notifications and audio feedback for recording/processing/complete/error |
| `config` | Single TOML config (`~/.config/chezwizper/config.toml`), one section per subsystem |
| `bench_trace` | Optional JSONL trace sink (`CHEZWIZPER_BENCH_TRACE`) emitting per-phase latency events |

## Control flow

### Command ingestion and the status state machine

The API server (`api/mod.rs`) and the main loop share a short-held `Arc<std::sync::Mutex<AppStatus>>` with three states: `Idle → Recording → Processing → Idle`. Status locks recover from poisoning and are never held across an `.await`. Handlers do not perform work; `reserve_recording_command` holds one synchronous critical section while it checks the current status, transitions it, and attempts to enqueue an `ApiCommand` on the bounded mpsc channel. If the enqueue fails, it rolls the transition back before releasing the lock. This gives:

- **Idempotent push-to-talk**: `/start` while recording and `/stop` while idle are accepted no-ops, so key-repeat and double-fires are harmless.
- **Rollback on backpressure**: if the channel is full, the reservation rolls the status back and returns 503 instead of leaving the state machine wedged.
- **Ordering**: a fast start→stop pair is queued in order even if the main loop hasn't consumed the start yet.

`/toggle` maps to start-or-stop based on the current state; a toggle during `Processing` is refused (`success: false`) rather than queued.

Endpoints: `POST /start`, `POST /stop`, `POST /toggle`, `GET /status` (plain or `?style=waybar`), `GET /model/status`, `POST /model/unload`, `POST /model/reload`.

### Recording lifecycle (main event loop)

`main.rs` runs a single sequential loop over `ApiCommand`s — only one recording/transcription cycle is in flight at a time.

**StartRecording:**
1. Show the recording indicator (hyprctl notification + start sound).
2. `AudioStreamManager::start_recording()` — clears the sample buffer and builds a cpal input stream (mono, 16 kHz — Whisper's native rate). The callback appends f32 samples into a shared `Arc<Mutex<Vec<f32>>>`.
3. `TranscriptionService::prepare()` — begins loading the model **while the user is speaking**, hiding model-load latency inside the recording window.
4. If `[chunking] enabled` and the provider supports it (whisper-rs only), spawn a `PauseChunkingSession` that watches the live buffer.

**StopRecording:**
1. Stop and drop the cpal stream; move the samples out of the shared buffer (`std::mem::take`, no copy). When chunking is active, also retain a snapshot for the session's final segment.
2. Run VAD. If no speech was detected, skip provider dispatch, temporary-file creation, and transcription entirely and report "No speech detected".
3. Otherwise pass the VAD-trimmed 16 kHz mono f32 samples to `TranscriptionService::transcribe_samples` — either directly or as a full fallback if finishing the chunking session fails. The raw final chunking snapshot remains separate from the trimmed fallback samples.
4. If `auto_paste` is on, inject the text; show completion or error via the indicator.
5. The per-command recovery envelope attempts `recording_complete()` on the provider (which may release the model — see below), then releases the `Processing` reservation. A scope guard restores status to `Idle` after successful work, a returned error, cancellation, or a caught unwind. Provider lifecycle errors and panics are contained, so they cannot prevent the guard from running; whenever the envelope completes, lifecycle completion is attempted before `Idle` becomes visible.

`transcribe_samples` accepts mono, 16 kHz f32 samples. whisper-rs consumes that slice directly. File-based providers inherit a collision-safe temporary-WAV adapter; its retention guard owns cleanup through provider completion.

## Transcription stack

Three layers, top down:

1. **`TranscriptionService`** (`transcription/mod.rs`) — the object the rest of the app holds. Wraps a `WhisperTranscriber` plus a `Normalizer` chosen to match the provider's output format (e.g. whisper.cpp emits `[00:00:00.000 --> ...]` timestamps that must be stripped; OpenAI CLI output needs different cleanup). All model-lifecycle calls (`prepare`, `unload_model`, `reload_model`, `model_status`, `recording_complete`) pass through it.
2. **`WhisperTranscriber`** (`whisper/mod.rs`) — resolves the `[whisper] provider` config value to a concrete provider, or auto-detects one (OpenAI CLI, then whisper.cpp CLI) when unset. Translates the flat `ProviderConfig` into provider-specific option structs.
3. **`TranscriptionProvider`** (`whisper/provider.rs`) — the trait each backend implements. `transcribe(path, language)`, `name`, and `is_available` are required. `transcribe_samples` has an object-safe default that writes a unique `chezwizper-*.wav`, delegates to the path method, and either deletes or retains the file according to `AudioFileRetention`. Lifecycle hooks (`prepare`, `unload_model`, `recording_complete`, `model_status`, `supports_chunking`) have no-op defaults, so subprocess/API providers stay trivial.

### Providers

| Provider | Mechanism | Notes |
|----------|-----------|-------|
| `whisper-rs` | In-process via whisper.cpp bindings | Consumes f32 samples directly; supports warm model retention, in-memory chunking, and `audio_ctx` tuning |
| `whisper-cpp` | Spawns `whisper-cli` | Uses the default sample-to-WAV adapter; model reloaded from disk each utterance |
| `openai-cli` | Spawns the `whisper` Python CLI | Uses the default sample-to-WAV adapter |
| `openai-api` | HTTPS to OpenAI's transcription API | Uses the default sample-to-WAV adapter; requires `api_key`; never auto-detected |

See [Adding Providers](./adding-providers.md) for the extension guide.

### whisper-rs model lifecycle

The in-process provider keeps a `WhisperContext` + `WhisperState` behind a mutex with an explicit state machine: `cold → loading → warm → idle-unloaded / error`. The policy is driven by `keep_warm_for_secs`:

- **`0` (default)**: cold at idle. `prepare()` loads the model at recording start (overlapping with speech); `recording_complete()` unloads it after each utterance, releasing ~166 MiB.
- **Positive value**: the model stays warm between recordings and a lazy idle timer unloads it after the configured duration (checked on the next `prepare`/`model_status` call — there is no background timer thread).
- The `/model/unload`, `/model/reload`, and `/model/status` endpoints expose manual control and observability.

`audio_ctx` controls the encoder context window: `auto` sizes it to the utterance length (clamped to a measured floor of 640, below which decoding destabilizes), a fixed integer pins it, and `0`/`"off"` uses the full 1500 window.

`behavior.delete_audio_files` controls retention only when the selected provider materializes a WAV through the default adapter. whisper-rs never creates a recording file, so setting this option to `false` does not force an archival WAV.

## VAD

`VadProcessor` runs once on the full recording at stop time. Engine selection (`[vad] engine`): `auto` prefers Silero (the whisper.cpp Silero v5 model, loaded from disk) and falls back to a simple peak/RMS amplitude gate if the model is unavailable; Silero failures at runtime also fall back to amplitude. The output is the trimmed sample range (with `pad_ms` padding) plus a `skipped` flag that short-circuits the whole transcription pipeline when nothing crosses the speech threshold.

## Pause-triggered chunking

An optional latency optimization (`[chunking] enabled`, whisper-rs only): instead of transcribing the whole utterance after stop, a background task polls the live recording buffer every 100 ms while you speak.

- **`PauseSegmenter`** scans 30 ms windows at a 10 ms hop with the amplitude gate. When it has seen at least `min_chunk_ms` of speech followed by `pause_ms` of silence, it emits a segment. Consecutive segments overlap by `overlap_ms` so words split by a boundary are captured twice.
- Each segment is passed to whisper-rs as an in-memory f32 slice and transcribed immediately while recording continues — so by the time you stop, most of the audio is already transcribed.
- On stop, the session receives the final buffer snapshot (with the VAD's authoritative speech end), transcribes the tail, and **stitches** the chunk transcripts: `stitch_transcripts` deduplicates the overlap by matching the longest common word sequence (case- and punctuation-insensitive) across each boundary.
- Any empty or failed chunk causes the session to fail. The stop handler retains the separate full VAD-trimmed sample vector and uses it for an in-memory full transcription fallback.

Design details and measured results are in the [Chunking Experiment](./chunking-experiment.md).

## Text injection

`TextInjector` picks one of two plans per transcript:

- **Type**: single-line text is typed directly with `wtype` (or `ydotool`), never touching the clipboard. Timeout scales with text length so long transcripts aren't killed mid-type.
- **Guarded paste**: multiline text is copied to the clipboard, pasted into the focused window, and the previous clipboard contents restored after a settle delay (`restore_clipboard`). Long-form pastes remain available briefly for consumers that read the clipboard late.

Direct typing falls back to guarded paste on failure or when no typing tool is installed. Clipboard access tries wl-clipboard, then xclip, then xsel. `[injection] force_method` pins a plan for debugging. All external tools are invoked with hard timeouts so a hung helper can't wedge the daemon. See [Text Injection Setup](./text-injection-setup.md).

## Observability and benchmarking

Setting `CHEZWIZPER_BENCH_TRACE=<path>` makes phase boundaries emit JSONL events (schema `chezwizper-bench-event-v1`) with monotonic and wall-clock timestamps: API receipt, state transitions, first audio sample, stream teardown, VAD trim, model load, provider transcription, normalization, and injection. `wav_write_begin`/`wav_write_done` are emitted only when a file-based provider actually materializes a WAV; direct whisper-rs full and chunked routes emit neither. When the variable is unset, the sink is disabled and event closures are never evaluated, so tracing costs nothing in production.

The harness under `bench/` plays a phrase corpus through a PipeWire loopback into a scratch daemon instance (on its own `[api] port`) and aggregates these traces into per-phase latency, RTF, and memory reports (`bench-artifacts/`). See [Benchmarking](./benchmarking.md).

## Configuration

One TOML file (`~/.config/chezwizper/config.toml`, overridable with `--config`) deserialized into `Config`, with a section per subsystem: `[audio]`, `[whisper]`, `[vad]`, `[chunking]`, `[ui]` (+ `[ui.waybar]`), `[wayland]`, `[behavior]`, `[injection]`, `[api]`. All sections have defaults; the full reference is in the [Configuration Guide](./configuration.md).

## Threading model

- The **main loop** is a single async task; recording cycles never overlap.
- Each **StopRecording command** has its own unwind boundary. Rust panics from stop work or provider lifecycle completion are logged and contained so the command consumer remains alive for the next request.
- **cpal** delivers audio on its own realtime callback thread, writing into the shared sample buffer.
- The **API server** runs as a spawned Tokio task; handlers only hold the synchronous status mutex for enum transitions and nonblocking command reservation.
- **Chunk transcriptions** run on `spawn_blocking` threads (whisper inference is CPU-bound and synchronous); the chunking watcher itself is an async task.
- Whisper-rs inference is serialized behind the provider's internal mutex, so concurrent chunk transcriptions queue rather than contend for the model.
