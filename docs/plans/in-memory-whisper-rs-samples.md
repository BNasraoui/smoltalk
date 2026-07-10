> Produced by Codex gpt-5.6-sol (xhigh reasoning) on 2026-07-10 for bead ChezWizper-v5j.

# Implementation plan: ChezWizper-v5j

## Design outcome

Make in-memory samples the primary recording-to-transcription interface:

```text
AudioStreamManager
  → VAD-trimmed Vec<f32>
  → TranscriptionService::transcribe_samples
  → WhisperTranscriber::transcribe_samples
  → provider.transcribe_samples
       whisper-rs: infer directly from &[f32]
       other providers: temporary WAV → existing transcribe(path)
```

Keep the existing path-based `transcribe()` API for compatibility. OpenAI API, OpenAI CLI, and whisper.cpp implementations require no changes.

Use an explicit retention enum rather than passing the configuration boolean through several inverted conditions:

```rust
enum AudioFileRetention {
    Delete,
    Keep,
}
```

For `whisper-rs`, both policies produce no file. `delete_audio_files = false` only preserves a WAV when the selected provider actually needs one.

## Implementation order

1. Add collision-safe temporary-file support

Files:

- `Cargo.toml`
- `Cargo.lock`

Add `tempfile` as a direct dependency. Although it is already transitively present in the lockfile, declaring it directly is necessary before using it.

Use `tempfile::Builder` with a `chezwizper-` prefix and `.wav` suffix. This replaces `/tmp/chezwizper_<unix_secs>.wav`, which can collide when multiple recordings stop within the same second or multiple daemon processes run concurrently.

Convert the `NamedTempFile` into a closed `TempPath` before opening it with `hound`; this retains collision-safe RAII deletion without keeping two handles open.

2. Return samples from the recorder instead of a path

File: `src/audio/mod.rs`

Change:

```rust
RecordedAudio::Speech(PathBuf)
```

to:

```rust
RecordedAudio::Speech(Vec<f32>)
```

Update:

- `AudioStreamManager::stop_recording`
- `AudioStreamManager::stop_recording_with_snapshot`
- `AudioStreamManager::stop_recording_inner`

Remove their `output_path` arguments. After VAD:

- Return `RecordedAudio::NoSpeech` exactly as today when `vad_output.skipped`.
- Otherwise return `vad_output.samples` without writing a file.
- Continue returning `FinalRecordingSnapshot` when chunking is active. Its raw samples and authoritative `speech_end` remain separate from the VAD-trimmed samples used by full fallback.
- Keep `write_samples_to_wav` in this module as the shared file-provider adapter.
- Remove the nonexistent output path from `audio_stop_begin`.
- Replace “saved audio” logging/errors with recording/VAD wording.

This preserves the existing no-speech short circuit: no provider call, temp-file creation, or WAV trace event occurs.

3. Add the trait-level sample entry point

File: `src/whisper/provider.rs`

Add `AudioFileRetention` and an object-safe method matching the existing boxed-future style:

```rust
fn transcribe_samples<'a>(
    &'a self,
    samples: &'a [f32],
    language: &'a str,
    retention: AudioFileRetention,
) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>
```

Its default implementation should:

1. Allocate a unique `.wav` path using `tempfile::Builder`.
2. Apply retention before transcription:
   - `Delete`: retain the `TempPath` guard across the entire future so success, error, and early return all clean up.
   - `Keep`: persist the path before writing so the file also remains after a provider error.
3. Call `write_samples_to_wav`.
4. Log the actual path for users who retain recordings.
5. Delegate to the provider’s existing `transcribe(path, language)` method.
6. Return the delegated result unchanged.

Only this default implementation invokes `write_samples_to_wav`, so `wav_write_begin/done` represent real writes.

4. Make whisper-rs consume the slice directly

File: `src/whisper/providers/whisper_rs.rs`

Refactor:

- Rename/rework `transcribe_loaded(audio_path, language)` into a slice-based helper such as `transcribe_loaded_samples(samples, language)`.
- Move only inference into that helper: build `FullParams` from `samples.len()`, call `WhisperState::full(params, samples)`, collect segments, and update warm-model state.
- Keep `read_wav_samples` for the existing path-based `transcribe()` method. That method must continue supporting f32 and i16 WAV callers.
- Override `transcribe_samples()` to:
  1. Ignore `AudioFileRetention`.
  2. Ensure the model is loaded.
  3. Call `transcribe_loaded_samples(samples, language)` directly.
- Make the existing `transcribe(path, language)` load the WAV and then call the same slice helper.

This ensures full and chunked sample calls cannot reach `hound::WavReader`, while preserving the public file path route.

No changes are required in:

- `src/whisper/providers/openai_api.rs`
- `src/whisper/providers/openai_cli.rs`
- `src/whisper/providers/whisper_cpp.rs`
- `src/whisper/providers/mod.rs`

They inherit the default adapter.

5. Thread samples through the transcription facade

File: `src/whisper/mod.rs`

Add:

```rust
WhisperTranscriber::transcribe_samples(
    &self,
    samples: &[f32],
    retention: AudioFileRetention,
) -> Result<String>
```

It should pass the configured language and retention to the provider.

Keep `WhisperTranscriber::transcribe(path)` unchanged for compatibility. Factor result tracing into a shared helper so each route emits exactly one `provider_transcription_begin` and one success/error end event.

For sample input, trace:

- `provider`
- `language`
- `input_kind: "samples"`
- `samples`

Do not attach a prospective `audio_path`; for whisper-rs no such file exists. Actual file providers are observable through `wav_write_*`, which include the generated path.

6. Thread samples through normalization

File: `src/transcription/mod.rs`

Add:

```rust
TranscriptionService::transcribe_samples(
    &self,
    samples: &[f32],
    retention: AudioFileRetention,
) -> Result<String>
```

Share normalization and end tracing between the path and sample methods. The sample method should emit:

- `transcription_begin` with `input_kind: "samples"` and sample count.
- `transcription_raw_done`
- `normalization_done`
- `transcription_end`

Keep `TranscriptionService::transcribe(path)` so library/path-based callers remain valid.

7. Remove per-chunk WAV handling

File: `src/chunking/mod.rs`

Update `transcribe_chunk`:

- Remove `write_samples_to_wav`.
- Remove `chunk_path`, `PathBuf`, and per-chunk file cleanup.
- Keep `spawn_blocking`, because whisper-rs inference remains synchronous and CPU-bound.
- Move the chunk `Vec<f32>` into the blocking closure and call:

```rust
runtime.block_on(
    transcription_service.transcribe_samples(
        &samples,
        AudioFileRetention::Delete,
    ),
)
```

`Delete` is appropriate for chunk intermediates and matches current unconditional chunk cleanup. It is ignored by whisper-rs, but safely handles any future file-based provider that advertises chunk support.

Keep `chunk_transcription_begin/end` around the entire operation with session ID, chunk index, sample count, and success. No `wav_write_*` event should appear for whisper-rs chunks.

8. Update the main stop and fallback paths

File: `src/main.rs`

Remove generation of `/tmp/chezwizper_<seconds>.wav`.

Call:

- `stop_recording()` without a path for normal recording.
- `stop_recording_with_snapshot()` without a path for chunking.

Match `RecordedAudio::Speech(samples)` and derive retention once:

```rust
let retention = if config.behavior.delete_audio_files {
    AudioFileRetention::Delete
} else {
    AudioFileRetention::Keep
};
```

For normal transcription, call `transcription_service.transcribe_samples(&samples, retention)`.

For chunking:

1. Keep the VAD-trimmed full `samples` alive while `session.finish(snapshot)` runs.
2. On chunking success, use its stitched text.
3. On any chunking failure, call `transcribe_samples(&samples, retention)`.

The fallback must use the VAD-trimmed full samples, not the raw chunking snapshot. This makes fallback independent of whether a file was created.

Remove main’s post-transcription `remove_file` block; file lifetime now belongs to the default trait adapter. Update the audio-stop error text so it no longer says “Failed to save audio.”

Consider extracting the finish-or-fallback branch into a private helper so its sample-based fallback can be unit-tested independently of the event loop.

9. Preserve meaningful benchmark semantics

Files:

- `bench/test_bench_harness.py`
- `docs/benchmarking.md`

Do not emit synthetic `wav_write_begin/done` events for the direct route.

The existing harness already tolerates a missing `wav_write_done`:

- `stop_to_wav_ms` becomes `null` for in-memory whisper-rs.
- Transcription duration remains populated from `transcription_begin` to `transcription_end`.
- `total_stop_to_text_ms` remains directly comparable and is the acceptance metric.

Add a harness regression assertion that a successful trial without any WAV event still has:

- status `ok`
- `stop_to_wav_ms is None`
- non-null transcription/RTF
- non-null `total_stop_to_text_ms`

Document that `stop_to_wav_ms` is provider-specific and absent for in-memory providers. The legacy `wav_to_transcription_ms` field should remain for artifact compatibility, even though it is calculated from transcription begin/end; clarify that it represents the transcription pipeline duration.

10. Update architecture documentation

File: `docs/architecture.md`

Update:

- The audio module description: capture, VAD, and sample return; conditional WAV encoding is used only by file-based providers.
- Stop flow: VAD-trimmed samples go to `transcribe_samples`.
- Trait description: path method plus default sample-to-WAV adapter.
- Provider table: whisper-rs consumes f32 samples directly.
- Chunking description: chunks are transcribed in memory.
- Fallback description: full VAD-trimmed samples are retained for re-transcription.
- `delete_audio_files`: controls retention only when a provider materializes a WAV.
- Trace documentation: WAV events are conditional.

## Tests

Rust tests should cover:

1. Default trait adapter:
   - Delegates to `transcribe`.
   - Produces mono, 16 kHz, 32-bit float WAV content matching the input samples.
   - Uses distinct paths across concurrent calls.
   - Deletes on success and error under `Delete`.
   - Preserves a valid WAV on success and error under `Keep`.

2. Whisper facade:
   - A fake provider overriding `transcribe_samples` receives the original slice, language, and retention.
   - Its path-based `transcribe` can panic in this test, proving the override route did not fall back to a WAV.

3. whisper-rs:
   - Existing path transcription still reads f32 and i16 WAV input.
   - `audio_ctx` continues using the direct slice length.
   - A model-backed integration smoke test transcribes an in-memory corpus sample without creating a `.wav`.

4. Audio:
   - Existing move/snapshot tests continue to prove the shared buffer is emptied.
   - Add coverage that speech results own the VAD-trimmed sample vector.
   - No-speech still returns `NoSpeech` without invoking the writer.

5. Chunking:
   - Extend worker tests to assert the exact chunk sample vectors received by the callback, not only their lengths.
   - Verify successful multi-chunk stitching.
   - Verify an empty/error chunk causes failure and the extracted main helper invokes full transcription with the retained trimmed samples.

6. Provider compatibility:
   - Existing whisper.cpp command tests continue passing unchanged.
   - Compilation of OpenAI CLI/API providers proves the default method requires no implementation changes.

Run:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
python3 -m unittest discover -s bench -p 'test_*.py'
```

## Benchmark verification

Use the same corpus, model, thread count, `audio_ctx`, VAD, machine state, and warm-trial count for baseline and candidate runs:

```bash
cargo build --release
python3 bench/make_corpus.py \
  --tts flite \
  --output-dir bench-artifacts/corpus

python3 bench/bench_e2e.py \
  --output-dir bench-artifacts/v5j-after-full \
  --provider whisper-rs \
  --model base.en
```

Repeat with a base config having chunking enabled:

```bash
python3 bench/bench_e2e.py \
  --base-config /path/to/chunking-enabled.toml \
  --output-dir bench-artifacts/v5j-after-chunked \
  --provider whisper-rs \
  --model base.en
```

Verify:

- Zero failures.
- Full and chunked traces contain no `wav_write_begin` or `wav_write_done`.
- Chunked traces contain paired successful `chunk_transcription_begin/end` events.
- `stop_to_wav_ms` is null for whisper-rs.
- `total_stop_to_text_ms` remains populated.
- Full and chunked stop-to-text p50/p95 do not regress against equivalent pre-change runs; use a 5% tolerance for run-to-run noise and rerun if CPU temperature or load differs materially.
- Peak RSS does not grow unexpectedly from retaining the full fallback vector.
- A file-provider smoke run still emits paired WAV events and transcribes successfully.
- File-provider runs delete the WAV under `Delete` and preserve a uniquely named valid WAV under `Keep`.

## Principal risks

- The `TempPath` guard must live until the delegated future completes; dropping it before a CLI/API provider finishes would remove its input.
- `Keep` must be applied before writing so failed transcriptions preserve the recording consistently.
- The sample-rate invariant is implicit in `&[f32]`: document that this entry point accepts 16 kHz mono samples, and keep all conversion at capture boundaries.
- Chunking already retains a raw final snapshot; retaining the separate trimmed vector for fallback temporarily keeps both allocations alive. Measure peak RSS on long recordings.
- File-provider transcription timing will now include its default-adapter WAV write inside the transcription pipeline interval. Compare total stop-to-text across the change and treat `stop_to_wav_ms` as an optional provider-specific phase.
- `delete_audio_files = false` no longer forces whisper-rs to create an archival WAV. That is intentional and required by the acceptance criteria, but must be documented clearly.
