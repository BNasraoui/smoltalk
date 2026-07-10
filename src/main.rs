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
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::api::{ApiCommand, ApiCommandSource, ApiServer, AppStatus};
use crate::audio::{AudioStreamManager, RecordedAudio};
use crate::chunking::PauseChunkingSession;
use crate::config::Config;
use crate::text_injection::TextInjector;
use crate::transcription::TranscriptionService;
use crate::ui::Indicator;
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
    status: Arc<Mutex<AppStatus>>,
    audio_recorder: Arc<Mutex<AudioStreamManager>>,
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
    if let Err(e) = transcription_service.prepare().await {
        error!("Transcription provider preload failed: {}", e);
    }

    let text_injector =
        TextInjector::new(Some(&config.wayland.input_method), config.injection.clone())?;

    let indicator =
        Indicator::from_config(&config.ui).with_audio_feedback(config.behavior.audio_feedback);

    let app_status = Arc::new(Mutex::new(AppStatus::Idle));
    let state = RecordingState {
        status: app_status.clone(),
        audio_recorder: Arc::new(Mutex::new(audio_recorder)),
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

                let audio_recorder = state.audio_recorder.lock().await;
                if let Err(e) = audio_recorder.start_recording().await {
                    error!("Failed to start recording: {}", e);
                    bench_trace::event_with_extra("trial_error", || {
                        serde_json::json!({
                            "phase": "audio_start",
                            "error": e.to_string(),
                        })
                    });
                    *state.status.lock().await = AppStatus::Idle;
                    bench_trace::event("state_idle_set");
                    let _ = indicator
                        .show_error(&format!("Recording failed: {e}"))
                        .await;
                } else if config.chunking.enabled && transcription_service.supports_chunking() {
                    chunking_session = Some(PauseChunkingSession::start(
                        &config.chunking,
                        &config.vad,
                        audio_recorder.sample_rate(),
                        audio_recorder.recording_buffer(),
                        transcription_service.clone(),
                    ));
                } else if config.chunking.enabled {
                    tracing::warn!("Pause chunking is only supported by the whisper-rs provider");
                }
            }
            ApiCommand::StopRecording(source) => {
                emit_dequeue_event(source);
                info!("Stopping recording");

                let temp_path = PathBuf::from(format!(
                    "/tmp/chezwizper_{}.wav",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ));

                let active_chunking_session = chunking_session.take();
                let (stop_result, final_samples) = {
                    let audio_recorder = state.audio_recorder.lock().await;
                    if active_chunking_session.is_some() {
                        match audio_recorder
                            .stop_recording_with_snapshot(temp_path.clone())
                            .await
                        {
                            Ok((audio, samples)) => (Ok(audio), Some(samples)),
                            Err(error) => (Err(error), None),
                        }
                    } else {
                        (audio_recorder.stop_recording(temp_path.clone()).await, None)
                    }
                };

                let mut active_chunking_session = active_chunking_session;
                let mut final_samples = final_samples;

                match stop_result {
                    Ok(RecordedAudio::Speech(audio_path)) => {
                        // Show processing indicator
                        if let Err(e) = indicator.show_processing().await {
                            error!("Failed to show processing indicator: {}", e);
                        }

                        let transcription_result = match (
                            active_chunking_session.take(),
                            final_samples.take(),
                        ) {
                            (Some(session), Some(samples)) => match session.finish(samples).await {
                                Ok(text) => Ok(text),
                                Err(error) => {
                                    tracing::warn!(
                                        "Pause chunking failed; falling back to full transcription: {error}"
                                    );
                                    transcription_service.transcribe(&audio_path).await
                                }
                            },
                            _ => transcription_service.transcribe(&audio_path).await,
                        };

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

                        // Clean up audio file
                        if config.behavior.delete_audio_files {
                            let _ = std::fs::remove_file(&audio_path);
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
                            .show_error(&format!("Failed to save audio: {e}"))
                            .await;
                    }
                }

                *state.status.lock().await = AppStatus::Idle;
                bench_trace::event("state_idle_set");
            }
        }
    }

    Ok(())
}

fn emit_dequeue_event(source: ApiCommandSource) {
    match source {
        ApiCommandSource::Start => bench_trace::event("main_start_dequeued"),
        ApiCommandSource::Stop => bench_trace::event("main_stop_dequeued"),
        ApiCommandSource::Toggle => bench_trace::event("main_toggle_dequeued"),
    }
}
