> Produced by Codex gpt-5.6-sol (xhigh reasoning) on 2026-07-10 for bead ChezWizper-bbj.

# Implementation plan for ChezWizper-bbj

Use deterministic Drop-based state restoration plus panic containment. Do not add a watchdog: valid transcription can take an unbounded amount of time, and resetting status while stop work is still running would expose a false `Idle` state, admit misleading commands, and eventually fill the queue if the main loop is genuinely hung.

## 1. Make `AppStatus` synchronously resettable

File: `src/api/mod.rs`

1. Replace the shared `tokio::sync::Mutex<AppStatus>` with a short-held `std::sync::Mutex<AppStatus>` and introduce a crate-visible `SharedAppStatus` alias.

   This is safe because every status critical section only reads or transitions the enum and calls nonblocking `try_send`; none needs to hold the lock across an `.await`.

2. Add a small `lock_app_status` helper that recovers poisoned mutexes with `PoisonError::into_inner`. Panic recovery must not replace the Processing wedge with a poisoned-lock wedge.

3. Add `ProcessingResetGuard` owning a cloned `SharedAppStatus`. Its `Drop` implementation must:

   - set the status unconditionally to `AppStatus::Idle`;
   - release the mutex before emitting `bench_trace::event("state_idle_set")`;
   - require no async runtime or spawned cleanup task.

4. Update these users to the new shared type and synchronous lock helper:

   - `AppState::status`
   - `ApiServer::new`
   - `reserve_recording_command`
   - `start_recording`
   - `stop_recording`
   - `toggle_recording`
   - `recording_status`

5. Preserve `reserve_recording_command`’s existing atomic critical section exactly: transition, `try_send`, and rollback remain under the same lock. Preserve all existing behavior:

   - start/toggle from Idle reserves Recording;
   - stop/toggle from Recording reserves Processing;
   - failed enqueue rolls Start back to Idle and Stop back to Recording;
   - toggle during Processing remains refused;
   - duplicate start/stop requests remain idempotent.

6. Mechanically update the existing tests’ status assertions from async locking to `lock_app_status`. Keep the queue-order and channel-full rollback tests unchanged in meaning.

## 2. Make the audio state recover from the same unwind

File: `src/audio/mod.rs`

1. Add a private `RecordingStateResetGuard` owning the audio state `Arc<Mutex<RecordingState>>`.

2. In `AudioStreamManager::stop_recording_inner`, create the guard immediately after transitioning `RecordingState::Recording → Stopping` and releasing the state lock.

3. Remove the three scattered assignments that restore `RecordingState::Idle` for empty samples, no speech, and WAV completion. Let the guard perform the single restoration on every return and panic path.

4. Add a device-independent unit test proving that dropping/unwinding the guard changes Stopping back to Idle.

This prevents the recovered API from accepting a new start only for `AudioStreamManager::start_recording()` to reject it as “Previous recording still stopping.”

## 3. Extract stop work and contain panics

Files: `Cargo.toml`, `Cargo.lock`, then `src/main.rs`

1. Add a direct `futures-util` dependency for `FutureExt::catch_unwind`; update the lockfile normally. `std::panic::catch_unwind` around future construction is insufficient because a panic can occur during a later poll.

2. Update `RecordingState::status` and initialization in `main` to use `SharedAppStatus`. Retain Tokio’s mutex, under a distinct import name, for `AudioStreamManager`.

3. Extract the current StopRecording body into a concrete async `handle_stop_recording` function. It should own the existing work from `emit_dequeue_event` through:

   - temp-path creation;
   - taking/finalizing/cancelling the chunking session;
   - audio stop and snapshot handling;
   - processing indicator;
   - transcription and fallback;
   - text injection;
   - completion/error indicator;
   - optional WAV deletion.

   It should not reset `AppStatus` or call `recording_complete()`.

4. Add a small testable lifecycle envelope, `run_stop_with_recovery`, which accepts the shared status, the stop-work future, and a synchronous lifecycle-completion callback.

   Its order must be:

   1. Construct `ProcessingResetGuard` before polling any stop work.
   2. Await the work using `AssertUnwindSafe(...).catch_unwind()`.
   3. Log a returned error or panic without rethrowing it.
   4. Invoke the supplied completion callback even after a returned error or caught panic.
   5. Protect that callback with synchronous `catch_unwind`; log either its `Err` or panic.
   6. Return normally, causing the status guard to restore Idle.

5. Replace the StopRecording match arm with a call to this envelope, passing `handle_stop_recording(...)` and a callback to `transcription_service.recording_complete()`.

This catch boundary is essential: a Drop guard restores the status during unwinding, but without catching the panic the command-consumer loop itself would die and could not process the next Start command.

## 4. Add regression tests in the existing Tokio style

Files: `src/main.rs` test module and the existing tests in `src/api/mod.rs`

Add focused tests using an mpsc channel and real `AppStatus`, without audio hardware or transcription mocks:

1. `mid_stop_error_resets_status_and_accepts_next_start`

   - Initialize status as Processing.
   - Run `run_stop_with_recovery` with a future that yields once and returns an injected error.
   - Assert the lifecycle callback ran once.
   - Assert status is Idle.
   - Reserve a Start request and assert it transitions to Recording and enqueues `StartRecording`.

2. `mid_stop_panic_resets_status_and_accepts_next_toggle`

   - Initialize status as Processing.
   - Use a future that yields once and then panics, ensuring the test exercises panic during async polling.
   - Assert the panic is contained rather than propagated from the envelope.
   - Assert the lifecycle callback ran once and status is Idle.
   - Reserve Toggle and assert it succeeds as a new Start command.

3. `recording_complete_failure_cannot_block_idle_reset`

   - Let stop work complete.
   - Make the completion callback return an error, and separately cover a callback panic if not folded into the panic test.
   - Assert the envelope returns and status becomes Idle.

4. Keep all existing `reserve_recording_command` tests as regression coverage for reservation, queue ordering, idempotence, and rollback.

## 5. Document the new ownership boundary

File: `docs/architecture.md`

Update:

- “Command ingestion and the status state machine” to describe the short synchronous status mutex and atomic reservation/rollback.
- “StopRecording” to state that a scope guard owns the Processing reservation and restores Idle after success, error, cancellation, or caught unwind.
- The lifecycle ordering to make clear that `recording_complete()` is attempted before the status becomes Idle.
- “Threading model” to note that Rust panics from stop handling are contained at the per-command boundary.

Do not describe a watchdog because none should be implemented.

## Risks and safeguards

- **Chunking cleanup:** Every expected `NoSpeech` and audio-stop error path must continue explicitly calling `session.cancel().await`; chunk-finalization failure must continue falling back to full transcription. On a panic, dropping `PauseChunkingSession` closes its command channel, so its worker exits when control returns to it, although an already-running blocking transcription cannot be forcibly stopped. Avoid introducing new `?` exits after `chunking_session.take()` unless cleanup has already been arranged.

- **Indicator errors:** Preserve the current best-effort handling. Indicator failures should be logged or ignored as today and must not become `?` exits that skip chunk cancellation, file cleanup, or provider lifecycle completion.

- **`recording_complete()`:** Keep it outside the extracted work but inside the recovery envelope so it is attempted after every work outcome. Its error or panic must never prevent the status guard from running. A provider panic may still leave provider-internal state unusable, but it must not kill the command loop or wedge `AppStatus`.

- **Audio consistency:** The audio-state guard prevents `RecordingState::Stopping` from surviving an unwind. A panic inside external audio or provider code can still leave deeper resources unhealthy, but the next request will be accepted and the normal start cleanup will drop any surviving stream and clear samples.

- **Mutex blocking:** The new standard mutex must remain limited to status reads, enum transitions, and `try_send`; never place an `.await` or slow operation inside its critical section.

- **Panic model:** Recovery applies to ordinary Rust unwinding, which is the repository’s current Cargo default. Abort panics, process termination, OOM aborts, and native crashes cannot run Drop or be recovered in-process.

## Verification

Run, in order:

1. `cargo fmt -- --check`
2. Targeted API, main-loop recovery, and audio-state tests
3. `cargo test --all-targets`
4. `cargo clippy --all-targets --all-features -- -D warnings`

No files or issue-tracker state were modified while producing this plan.
