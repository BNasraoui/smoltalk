use anyhow::Result;
use futures_util::FutureExt;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info};

use crate::api::{
    release_recording_reservation, ApiCommandSource, ApiServer, AppLifecycle, ProcessingResetGuard,
    RecordingCommand, SharedAppLifecycle,
};
use crate::audio::{AudioStreamManager, RecordedAudio};
use crate::cancellation::CancellationToken;
use crate::chunking::PauseChunkingSession;
use crate::config::Config;
use crate::text_injection::TextInjector;
use crate::transcription::TranscriptionService;
use crate::ui::Indicator;
use crate::whisper::provider::AudioFileRetention;
use crate::whisper::WhisperTranscriber;
use crate::{bench_trace, whisper};

struct Daemon {
    lifecycle: SharedAppLifecycle,
    audio_recorder: AudioStreamManager,
    transcription_service: Arc<TranscriptionService>,
    text_injector: TextInjector,
    indicator: Indicator,
    config: Config,
    chunking_session: Option<PauseChunkingSession>,
}

pub async fn run(config_path: Option<PathBuf>) -> Result<()> {
    info!("Starting ChezWizper");
    bench_trace::event("daemon_start");

    // Load configuration
    let config = if let Some(config_path) = config_path {
        Config::load_from_path(config_path)?
    } else {
        Config::load()?
    };
    // Initialize components
    let (tx, rx) = mpsc::channel::<RecordingCommand>(10);

    let audio_recorder =
        AudioStreamManager::new_with_vad(&config.audio.device, config.vad.clone())?;

    // Build whisper transcriber
    let whisper = if let Some(provider) = &config.whisper.provider {
        let provider_config = whisper::ProviderConfig {
            model: Some(config.whisper.model.clone()),
            model_path: config.whisper.model_path.clone(),
            language: Some(config.whisper.language.clone()),
            command_path: config.whisper.command_path.clone(),
            api_endpoint: config.whisper.api_endpoint.clone(),
            api_key: config.whisper.api_key.clone(),
            threads: config.whisper.threads,
            beam_size: config.whisper.beam_size,
            best_of: config.whisper.best_of,
            no_fallback: config.whisper.no_fallback,
            timeout_secs: config.whisper.timeout_secs,
            keep_warm_for_secs: config.whisper.keep_warm_for_secs,
            initial_prompt: config.whisper.initial_prompt.clone(),
            coding_vocabulary: config.whisper.coding_vocabulary.clone(),
            audio_ctx: config.whisper.audio_ctx,
        };
        WhisperTranscriber::with_provider(provider, provider_config)?
    } else {
        // Auto-detect provider when no provider specified
        let provider_config = whisper::ProviderConfig {
            model: Some(config.whisper.model.clone()),
            model_path: config.whisper.model_path.clone(),
            language: Some(config.whisper.language.clone()),
            command_path: config.whisper.command_path.clone(),
            api_endpoint: config.whisper.api_endpoint.clone(),
            api_key: config.whisper.api_key.clone(),
            threads: config.whisper.threads,
            beam_size: config.whisper.beam_size,
            best_of: config.whisper.best_of,
            no_fallback: config.whisper.no_fallback,
            timeout_secs: config.whisper.timeout_secs,
            keep_warm_for_secs: config.whisper.keep_warm_for_secs,
            initial_prompt: config.whisper.initial_prompt.clone(),
            coding_vocabulary: config.whisper.coding_vocabulary.clone(),
            audio_ctx: config.whisper.audio_ctx,
        };
        WhisperTranscriber::auto_detect(provider_config)?
    };

    // Compose transcription service with whisper and normalizer
    let transcription_service = Arc::new(TranscriptionService::new(whisper)?);
    if config.whisper.keep_warm_for_secs != Some(0) {
        if let Err(e) = transcription_service.prepare().await {
            error!("Transcription provider preload failed: {}", e);
        }
    }

    let text_injector =
        TextInjector::new(Some(&config.wayland.input_method), config.injection.clone())?;

    let indicator =
        Indicator::from_config(&config.ui).with_audio_feedback(config.behavior.audio_feedback);

    let lifecycle = Arc::new(std::sync::Mutex::new(AppLifecycle::Idle));
    // Create and start API server
    let api_server = ApiServer::new(
        tx,
        lifecycle.clone(),
        &config,
        transcription_service.clone(),
    );

    // Start API server in background
    tokio::spawn(async move {
        if let Err(e) = api_server.start().await {
            error!("API server failed: {}", e);
        }
    });

    // Print instructions for Hyprland setup
    info!("ChezWizper is ready!");
    bench_trace::event("daemon_ready");
    info!("Add this to your Hyprland config:");
    info!("bindd = SUPER, R, ChezWizper, exec, curl -X POST http://127.0.0.1:3737/toggle");
    info!("Or test manually: curl -X POST http://127.0.0.1:3737/toggle");

    Daemon {
        lifecycle,
        audio_recorder,
        transcription_service,
        text_injector,
        indicator,
        config,
        chunking_session: None,
    }
    .run(rx)
    .await
}

impl Daemon {
    async fn run(mut self, mut rx: mpsc::Receiver<RecordingCommand>) -> Result<()> {
        while let Some(command) = rx.recv().await {
            match command {
                RecordingCommand::Start(source, cancellation) => {
                    emit_dequeue_event(source);
                    if cancellation.is_cancelled() {
                        info!("Skipping cancelled recording start");
                        continue;
                    }
                    info!("Starting recording");

                    if let Err(e) = self.indicator.show_recording().await {
                        error!("Failed to show recording indicator: {}", e);
                    }

                    let start_result = { self.audio_recorder.start_recording().await };
                    if let Err(e) = start_result {
                        error!("Failed to start recording: {}", e);
                        bench_trace::event_with_extra("trial_error", || {
                            serde_json::json!({
                                "phase": "audio_start",
                                "error": e.to_string(),
                            })
                        });
                        release_recording_reservation(&self.lifecycle, &cancellation);
                        let _ = self
                            .indicator
                            .show_error(&format!("Recording failed: {e}"))
                            .await;
                    } else {
                        if let Err(e) = self.transcription_service.prepare().await {
                            error!("Transcription provider preparation failed: {}", e);
                        }

                        if cancellation.is_cancelled() {
                            info!("Recording was cancelled during provider preparation");
                            continue;
                        }

                        if self.config.chunking.enabled
                            && self.transcription_service.supports_chunking()
                        {
                            self.chunking_session = Some(PauseChunkingSession::start(
                                &self.config.chunking,
                                &self.config.vad,
                                self.audio_recorder.sample_rate(),
                                self.audio_recorder.recording_buffer(),
                                self.transcription_service.clone(),
                                cancellation.clone(),
                            ));
                        } else if self.config.chunking.enabled {
                            tracing::warn!(
                                "Pause chunking is only supported by the whisper-rs provider"
                            );
                        }
                    }
                }
                RecordingCommand::Stop(source, cancellation) => {
                    let transcription_service = self.transcription_service.clone();
                    run_stop_with_recovery(
                        self.lifecycle.clone(),
                        cancellation.clone(),
                        handle_stop_recording(&mut self, source, cancellation),
                        move || transcription_service.recording_complete(),
                    )
                    .await;
                }
                RecordingCommand::Cancel(cancellation) => {
                    bench_trace::event("main_cancel_dequeued");
                    let transcription_service = self.transcription_service.clone();
                    run_stop_with_recovery(
                        self.lifecycle.clone(),
                        cancellation,
                        handle_cancel_recording(&mut self),
                        move || transcription_service.recording_complete(),
                    )
                    .await;
                }
            }
        }

        Ok(())
    }
}

async fn handle_stop_recording(
    daemon: &mut Daemon,
    source: ApiCommandSource,
    cancellation: CancellationToken,
) -> Result<()> {
    emit_dequeue_event(source);
    info!("Stopping recording");

    if cancellation.is_cancelled() {
        if let Some(session) = daemon.chunking_session.take() {
            session.cancel().await;
        }
        daemon.audio_recorder.cancel_recording();
        let _ = daemon.indicator.show_cancelled().await;
        return Ok(());
    }

    let active_chunking_session = daemon.chunking_session.take();
    let (stop_result, final_samples) = {
        if active_chunking_session.is_some() {
            match daemon.audio_recorder.stop_recording_with_snapshot().await {
                Ok((audio, samples)) => (Ok(audio), Some(samples)),
                Err(error) => (Err(error), None),
            }
        } else {
            (daemon.audio_recorder.stop_recording().await, None)
        }
    };

    let mut active_chunking_session = active_chunking_session;
    let mut final_samples = final_samples;

    if cancellation.is_cancelled() {
        if let Some(session) = active_chunking_session.take() {
            session.cancel().await;
        }
        let _ = daemon.indicator.show_cancelled().await;
        return Ok(());
    }

    let retention = if daemon.config.behavior.delete_audio_files {
        AudioFileRetention::Delete
    } else {
        AudioFileRetention::Keep
    };

    match stop_result {
        Ok(RecordedAudio::Speech(samples)) => {
            // Show processing indicator
            if let Err(e) = daemon.indicator.show_processing().await {
                error!("Failed to show processing indicator: {}", e);
            }

            let chunking_finish = match (active_chunking_session.take(), final_samples.take()) {
                (Some(session), Some(snapshot)) => Some(session.finish(snapshot)),
                _ => None,
            };
            let transcription_result = finish_chunking_or_fallback(
                chunking_finish,
                daemon.transcription_service.transcribe_samples(
                    &samples,
                    retention,
                    cancellation.clone(),
                ),
            )
            .await;

            // Transcribe audio
            match transcription_result {
                Ok(text) => {
                    if !text.is_empty() {
                        let delivered = deliver_if_active(&cancellation, || async {
                            info!("Transcription successful: {} chars", text.len());

                            if daemon.config.behavior.auto_paste {
                                if let Err(e) = daemon.text_injector.inject_text(&text).await {
                                    error!("Failed to inject text: {}", e);
                                    let _ = daemon
                                        .indicator
                                        .show_error(&format!(
                                            "Text injection failed: {e}. Transcript may be on clipboard if paste fallback copied it."
                                        ))
                                        .await;
                                }
                            }

                            if let Err(e) = daemon.indicator.show_complete(&text).await {
                                error!("Failed to show completion indicator: {}", e);
                            }
                        })
                        .await;

                        if delivered.is_none() {
                            let _ = daemon.indicator.show_cancelled().await;
                        }
                    } else {
                        show_terminal_feedback(
                            &cancellation,
                            || async {
                                let _ = daemon.indicator.show_error("No speech detected").await;
                            },
                            || async {
                                let _ = daemon.indicator.show_cancelled().await;
                            },
                        )
                        .await;
                    }
                }
                Err(e) => {
                    if cancellation.is_cancelled() {
                        info!("Transcription cancelled");
                        let _ = daemon.indicator.show_cancelled().await;
                        return Ok(());
                    }
                    show_terminal_feedback(
                        &cancellation,
                        || async {
                            error!("Transcription failed: {}", e);
                            bench_trace::event_with_extra("trial_error", || {
                                serde_json::json!({
                                    "phase": "transcription",
                                    "error": e.to_string(),
                                })
                            });
                            let _ = daemon
                                .indicator
                                .show_error(&format!("Transcription failed: {e}"))
                                .await;
                        },
                        || async {
                            let _ = daemon.indicator.show_cancelled().await;
                        },
                    )
                    .await;
                }
            }
        }
        Ok(RecordedAudio::NoSpeech) => {
            if let Some(session) = active_chunking_session.take() {
                session.cancel().await;
            }
            bench_trace::event_with_extra("transcription_skipped", || {
                serde_json::json!({
                    "reason": "no_speech",
                })
            });
            show_terminal_feedback(
                &cancellation,
                || async {
                    let _ = daemon.indicator.show_error("No speech detected").await;
                },
                || async {
                    let _ = daemon.indicator.show_cancelled().await;
                },
            )
            .await;
        }
        Err(e) => {
            if let Some(session) = active_chunking_session.take() {
                session.cancel().await;
            }
            show_terminal_feedback(
                &cancellation,
                || async {
                    error!("Failed to stop recording: {}", e);
                    bench_trace::event_with_extra("trial_error", || {
                        serde_json::json!({
                            "phase": "audio_stop",
                            "error": e.to_string(),
                        })
                    });
                    let _ = daemon
                        .indicator
                        .show_error(&format!("Failed to stop recording: {e}"))
                        .await;
                },
                || async {
                    let _ = daemon.indicator.show_cancelled().await;
                },
            )
            .await;
        }
    }

    Ok(())
}

async fn handle_cancel_recording(daemon: &mut Daemon) -> Result<()> {
    info!("Cancelling recording");
    if let Some(session) = daemon.chunking_session.take() {
        session.cancel().await;
    }
    daemon.audio_recorder.cancel_recording();
    let _ = daemon.indicator.show_cancelled().await;
    Ok(())
}

async fn deliver_if_active<F, Fut, T>(cancellation: &CancellationToken, deliver: F) -> Option<T>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    if !cancellation.try_commit() {
        return None;
    }

    Some(deliver().await)
}

async fn show_terminal_feedback<F, Fut, C, CancelFut>(
    cancellation: &CancellationToken,
    show_active: F,
    show_cancelled: C,
) where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
    C: FnOnce() -> CancelFut,
    CancelFut: Future<Output = ()>,
{
    if cancellation.is_cancelled() {
        show_cancelled().await;
        return;
    }

    show_active().await;

    if cancellation.is_cancelled() {
        show_cancelled().await;
    }
}

async fn finish_chunking_or_fallback<C, F>(
    chunking_finish: Option<C>,
    full_transcription: F,
) -> Result<String>
where
    C: Future<Output = Result<String>>,
    F: Future<Output = Result<String>>,
{
    if let Some(chunking_finish) = chunking_finish {
        match chunking_finish.await {
            Ok(text) => return Ok(text),
            Err(error) => {
                tracing::warn!("Pause chunking failed; falling back to full transcription: {error}")
            }
        }
    }

    full_transcription.await
}

async fn run_stop_with_recovery<F, C>(
    lifecycle: SharedAppLifecycle,
    cancellation: CancellationToken,
    stop_work: F,
    recording_complete: C,
) where
    F: Future<Output = Result<()>>,
    C: FnOnce() -> Result<()>,
{
    let _status_reset_guard = ProcessingResetGuard::new(lifecycle, cancellation);

    match AssertUnwindSafe(stop_work).catch_unwind().await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => error!("Stop recording handling failed: {}", e),
        Err(_) => error!("Stop recording handling panicked"),
    }

    match catch_unwind(AssertUnwindSafe(recording_complete)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => error!("Failed to complete transcription provider lifecycle: {}", e),
        Err(_) => error!("Transcription provider lifecycle panicked"),
    }
}

fn emit_dequeue_event(source: ApiCommandSource) {
    match source {
        ApiCommandSource::Start => bench_trace::event("main_start_dequeued"),
        ApiCommandSource::Stop => bench_trace::event("main_stop_dequeued"),
        ApiCommandSource::Toggle => bench_trace::event("main_toggle_dequeued"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{lock_lifecycle, reserve_recording_command, AppStatus, RecordingRequest};
    use crate::cancellation::CancellationToken;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    async fn panic_during_stop_poll() -> Result<()> {
        tokio::task::yield_now().await;
        panic!("injected stop panic");
    }

    #[tokio::test]
    async fn mid_stop_error_resets_status_and_accepts_next_start() {
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(std::sync::Mutex::new(AppLifecycle::Processing(
            cancellation.clone(),
        )));
        let completion_calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = completion_calls.clone();

        run_stop_with_recovery(
            lifecycle.clone(),
            cancellation,
            async {
                tokio::task::yield_now().await;
                Err(anyhow::anyhow!("injected stop failure"))
            },
            move || {
                callback_calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .await;

        assert_eq!(completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);

        let (tx, mut rx) = mpsc::channel(1);
        reserve_recording_command(&tx, &lifecycle, RecordingRequest::Start)
            .await
            .expect("start should be accepted after stop recovery");

        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Recording);
        assert!(matches!(
            rx.recv().await,
            Some(RecordingCommand::Start(ApiCommandSource::Start, _))
        ));
    }

    #[tokio::test]
    async fn mid_stop_panic_resets_status_and_accepts_next_toggle() {
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(std::sync::Mutex::new(AppLifecycle::Processing(
            cancellation.clone(),
        )));
        let completion_calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = completion_calls.clone();

        run_stop_with_recovery(
            lifecycle.clone(),
            cancellation,
            panic_during_stop_poll(),
            move || {
                callback_calls.fetch_add(1, Ordering::SeqCst);
                panic!("injected recording_complete panic");
            },
        )
        .await;

        assert_eq!(completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);

        let (tx, mut rx) = mpsc::channel(1);
        reserve_recording_command(&tx, &lifecycle, RecordingRequest::Toggle)
            .await
            .expect("toggle should be accepted after stop recovery");

        assert_eq!(lock_lifecycle(&lifecycle).status(), AppStatus::Recording);
        assert!(matches!(
            rx.recv().await,
            Some(RecordingCommand::Start(ApiCommandSource::Toggle, _))
        ));
    }

    #[tokio::test]
    async fn recording_complete_failure_cannot_block_idle_reset() {
        let cancellation = CancellationToken::new();
        let lifecycle = Arc::new(std::sync::Mutex::new(AppLifecycle::Processing(
            cancellation.clone(),
        )));
        let completion_calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = completion_calls.clone();

        run_stop_with_recovery(
            lifecycle.clone(),
            cancellation,
            async { Ok(()) },
            move || {
                callback_calls.fetch_add(1, Ordering::SeqCst);
                Err(anyhow::anyhow!("injected recording_complete failure"))
            },
        )
        .await;

        assert_eq!(completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*lock_lifecycle(&lifecycle), AppLifecycle::Idle);
    }

    #[tokio::test]
    async fn chunk_failure_falls_back_to_the_retained_trimmed_samples() {
        let trimmed_samples = vec![0.125, -0.25, 0.5];
        let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let fallback_observed = observed.clone();
        let fallback_samples = trimmed_samples.clone();

        let text = finish_chunking_or_fallback(
            Some(async { Err(anyhow::anyhow!("injected chunk failure")) }),
            async move {
                *fallback_observed.lock().unwrap() = fallback_samples;
                Ok("full fallback transcript".to_string())
            },
        )
        .await
        .unwrap();

        assert_eq!(text, "full fallback transcript");
        assert_eq!(*observed.lock().unwrap(), trimmed_samples);
    }

    #[tokio::test]
    async fn cancelled_transcript_does_not_run_delivery() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let called = Arc::new(AtomicBool::new(false));
        let delivery_called = called.clone();

        let delivered = deliver_if_active(&cancellation, move || async move {
            delivery_called.store(true, Ordering::SeqCst);
        })
        .await;

        assert!(delivered.is_none());
        assert!(!called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn cancellation_overrides_terminal_feedback_that_is_already_running() {
        let cancellation = CancellationToken::new();
        let feedback = Arc::new(std::sync::Mutex::new(Vec::new()));
        let active_feedback = feedback.clone();
        let cancelled_feedback = feedback.clone();
        let cancellation_during_feedback = cancellation.clone();

        show_terminal_feedback(
            &cancellation,
            move || async move {
                active_feedback.lock().unwrap().push("no speech");
                cancellation_during_feedback.cancel();
            },
            move || async move {
                cancelled_feedback.lock().unwrap().push("cancelled");
            },
        )
        .await;

        assert_eq!(*feedback.lock().unwrap(), ["no speech", "cancelled"]);
    }
}
