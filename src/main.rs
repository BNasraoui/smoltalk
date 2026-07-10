#![allow(clippy::arc_with_non_send_sync)]

mod api;
mod audio;
mod bench_trace;
mod chunking;
mod config;
mod normalizer;
mod text_injection;
mod transcription;
mod ui;
mod vad;
mod whisper;

use anyhow::Result;
use clap::Parser;
use futures_util::FutureExt;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex as TokioMutex};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::api::{
    lock_app_status, ApiCommand, ApiCommandSource, ApiServer, AppStatus, ProcessingResetGuard,
    SharedAppStatus,
};
use crate::audio::{AudioStreamManager, RecordedAudio};
use crate::chunking::PauseChunkingSession;
use crate::config::Config;
use crate::text_injection::TextInjector;
use crate::transcription::TranscriptionService;
use crate::ui::Indicator;
use crate::whisper::provider::AudioFileRetention;
use crate::whisper::WhisperTranscriber;

#[derive(Parser)]
#[command(name = "chezwizper")]
#[command(about = "Voice transcription tool for Wayland/Hyprland", long_about = None)]
struct Args {
    #[arg(short, long)]
    config: Option<PathBuf>,

    #[arg(short, long)]
    verbose: bool,
}

#[derive(Clone)]
struct RecordingState {
    status: SharedAppStatus,
    audio_recorder: Arc<TokioMutex<AudioStreamManager>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    let env_filter = EnvFilter::try_new(log_level).unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    info!("Starting ChezWizper");
    bench_trace::event("daemon_start");

    // Load configuration
    let config = if let Some(config_path) = args.config {
        Config::load_from_path(config_path)?
    } else {
        Config::load()?
    };
    // Initialize components
    let (tx, mut rx) = mpsc::channel::<ApiCommand>(10);

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

    let app_status = Arc::new(std::sync::Mutex::new(AppStatus::Idle));
    let state = RecordingState {
        status: app_status.clone(),
        audio_recorder: Arc::new(TokioMutex::new(audio_recorder)),
    };

    // Create and start API server
    let api_server = ApiServer::new(
        tx,
        app_status.clone(),
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

    let mut chunking_session: Option<PauseChunkingSession> = None;

    // Main event loop
    while let Some(command) = rx.recv().await {
        match command {
            ApiCommand::StartRecording(source) => {
                emit_dequeue_event(source);
                info!("Starting recording");

                if let Err(e) = indicator.show_recording().await {
                    error!("Failed to show recording indicator: {}", e);
                }

                let start_result = {
                    let audio_recorder = state.audio_recorder.lock().await;
                    audio_recorder.start_recording().await
                };
                if let Err(e) = start_result {
                    error!("Failed to start recording: {}", e);
                    bench_trace::event_with_extra("trial_error", || {
                        serde_json::json!({
                            "phase": "audio_start",
                            "error": e.to_string(),
                        })
                    });
                    *lock_app_status(&state.status) = AppStatus::Idle;
                    bench_trace::event("state_idle_set");
                    let _ = indicator
                        .show_error(&format!("Recording failed: {e}"))
                        .await;
                } else {
                    if let Err(e) = transcription_service.prepare().await {
                        error!("Transcription provider preparation failed: {}", e);
                    }

                    if config.chunking.enabled && transcription_service.supports_chunking() {
                        let audio_recorder = state.audio_recorder.lock().await;
                        chunking_session = Some(PauseChunkingSession::start(
                            &config.chunking,
                            &config.vad,
                            audio_recorder.sample_rate(),
                            audio_recorder.recording_buffer(),
                            transcription_service.clone(),
                        ));
                    } else if config.chunking.enabled {
                        tracing::warn!(
                            "Pause chunking is only supported by the whisper-rs provider"
                        );
                    }
                }
            }
            ApiCommand::StopRecording(source) => {
                run_stop_with_recovery(
                    state.status.clone(),
                    handle_stop_recording(
                        source,
                        &state.audio_recorder,
                        &mut chunking_session,
                        &indicator,
                        &transcription_service,
                        &text_injector,
                        &config,
                    ),
                    || transcription_service.recording_complete(),
                )
                .await;
            }
        }
    }

    Ok(())
}

async fn handle_stop_recording(
    source: ApiCommandSource,
    audio_recorder: &Arc<TokioMutex<AudioStreamManager>>,
    chunking_session: &mut Option<PauseChunkingSession>,
    indicator: &Indicator,
    transcription_service: &TranscriptionService,
    text_injector: &TextInjector,
    config: &Config,
) -> Result<()> {
    emit_dequeue_event(source);
    info!("Stopping recording");

    let active_chunking_session = chunking_session.take();
    let (stop_result, final_samples) = {
        let audio_recorder = audio_recorder.lock().await;
        if active_chunking_session.is_some() {
            match audio_recorder.stop_recording_with_snapshot().await {
                Ok((audio, samples)) => (Ok(audio), Some(samples)),
                Err(error) => (Err(error), None),
            }
        } else {
            (audio_recorder.stop_recording().await, None)
        }
    };

    let mut active_chunking_session = active_chunking_session;
    let mut final_samples = final_samples;
    let retention = if config.behavior.delete_audio_files {
        AudioFileRetention::Delete
    } else {
        AudioFileRetention::Keep
    };

    match stop_result {
        Ok(RecordedAudio::Speech(samples)) => {
            // Show processing indicator
            if let Err(e) = indicator.show_processing().await {
                error!("Failed to show processing indicator: {}", e);
            }

            let chunking_finish = match (active_chunking_session.take(), final_samples.take()) {
                (Some(session), Some(snapshot)) => Some(session.finish(snapshot)),
                _ => None,
            };
            let transcription_result = finish_chunking_or_fallback(
                chunking_finish,
                transcription_service.transcribe_samples(&samples, retention),
            )
            .await;

            // Transcribe audio
            match transcription_result {
                Ok(text) => {
                    if !text.is_empty() {
                        info!("Transcription successful: {} chars", text.len());

                        if config.behavior.auto_paste {
                            if let Err(e) = text_injector.inject_text(&text).await {
                                error!("Failed to inject text: {}", e);
                                let _ = indicator
                                    .show_error(&format!(
                                        "Text injection failed: {e}. Transcript may be on clipboard if paste fallback copied it."
                                    ))
                                    .await;
                            }
                        }

                        // Show completion
                        if let Err(e) = indicator.show_complete(&text).await {
                            error!("Failed to show completion indicator: {}", e);
                        }
                    } else {
                        let _ = indicator.show_error("No speech detected").await;
                    }
                }
                Err(e) => {
                    error!("Transcription failed: {}", e);
                    bench_trace::event_with_extra("trial_error", || {
                        serde_json::json!({
                            "phase": "transcription",
                            "error": e.to_string(),
                        })
                    });
                    let _ = indicator
                        .show_error(&format!("Transcription failed: {e}"))
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
            let _ = indicator.show_error("No speech detected").await;
        }
        Err(e) => {
            if let Some(session) = active_chunking_session.take() {
                session.cancel().await;
            }
            error!("Failed to stop recording: {}", e);
            bench_trace::event_with_extra("trial_error", || {
                serde_json::json!({
                    "phase": "audio_stop",
                    "error": e.to_string(),
                })
            });
            let _ = indicator
                .show_error(&format!("Failed to stop recording: {e}"))
                .await;
        }
    }

    Ok(())
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

async fn run_stop_with_recovery<F, C>(status: SharedAppStatus, stop_work: F, recording_complete: C)
where
    F: Future<Output = Result<()>>,
    C: FnOnce() -> Result<()>,
{
    let _status_reset_guard = ProcessingResetGuard::new(status);

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
    use crate::api::{reserve_recording_command, RecordingRequest};
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn panic_during_stop_poll() -> Result<()> {
        tokio::task::yield_now().await;
        panic!("injected stop panic");
    }

    #[tokio::test]
    async fn mid_stop_error_resets_status_and_accepts_next_start() {
        let status = Arc::new(std::sync::Mutex::new(AppStatus::Processing));
        let completion_calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = completion_calls.clone();

        run_stop_with_recovery(
            status.clone(),
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
        assert_eq!(*lock_app_status(&status), AppStatus::Idle);

        let (tx, mut rx) = mpsc::channel(1);
        reserve_recording_command(&tx, &status, RecordingRequest::Start)
            .await
            .expect("start should be accepted after stop recovery");

        assert_eq!(*lock_app_status(&status), AppStatus::Recording);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StartRecording(ApiCommandSource::Start))
        );
    }

    #[tokio::test]
    async fn mid_stop_panic_resets_status_and_accepts_next_toggle() {
        let status = Arc::new(std::sync::Mutex::new(AppStatus::Processing));
        let completion_calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = completion_calls.clone();

        run_stop_with_recovery(status.clone(), panic_during_stop_poll(), move || {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            panic!("injected recording_complete panic");
        })
        .await;

        assert_eq!(completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*lock_app_status(&status), AppStatus::Idle);

        let (tx, mut rx) = mpsc::channel(1);
        reserve_recording_command(&tx, &status, RecordingRequest::Toggle)
            .await
            .expect("toggle should be accepted after stop recovery");

        assert_eq!(*lock_app_status(&status), AppStatus::Recording);
        assert_eq!(
            rx.recv().await,
            Some(ApiCommand::StartRecording(ApiCommandSource::Toggle))
        );
    }

    #[tokio::test]
    async fn recording_complete_failure_cannot_block_idle_reset() {
        let status = Arc::new(std::sync::Mutex::new(AppStatus::Processing));
        let completion_calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = completion_calls.clone();

        run_stop_with_recovery(status.clone(), async { Ok(()) }, move || {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("injected recording_complete failure"))
        })
        .await;

        assert_eq!(completion_calls.load(Ordering::SeqCst), 1);
        assert_eq!(*lock_app_status(&status), AppStatus::Idle);
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
}
