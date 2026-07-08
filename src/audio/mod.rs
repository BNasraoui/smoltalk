#![allow(clippy::arc_with_non_send_sync)]

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info};

use crate::bench_trace;
use crate::vad::{VadOutput, VadProcessor, VadSettings};

/// State of the audio recording session
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecordingState {
    Idle,
    Recording,
    Stopping,
}

/// Manages the lifecycle of audio streams and recordings
pub struct AudioStreamManager {
    device: cpal::Device,
    config: cpal::StreamConfig,
    samples: Arc<Mutex<Vec<f32>>>,
    active_stream: Arc<Mutex<Option<cpal::Stream>>>,
    state: Arc<Mutex<RecordingState>>,
    first_sample_seen: Arc<AtomicBool>,
    vad: VadProcessor,
}

pub enum RecordedAudio {
    Speech(PathBuf),
    NoSpeech,
}

/// Resolve the configured `[audio] device` name to a cpal input device.
/// An unknown name warns and falls back to the system default rather than
/// failing: dictation must keep working after a device disappears.
fn select_input_device(host: &cpal::Host, device_name: &str) -> Result<cpal::Device> {
    if device_name != "default" {
        match host.input_devices() {
            Ok(mut devices) => {
                if let Some(device) =
                    devices.find(|d| d.name().map(|n| n == device_name).unwrap_or(false))
                {
                    return Ok(device);
                }
                tracing::warn!(
                    "Configured audio device \"{device_name}\" not found; falling back to default input"
                );
            }
            Err(e) => tracing::warn!("Could not enumerate input devices: {e}; using default"),
        }
    }

    host.default_input_device()
        .context("No input device available")
}

impl AudioStreamManager {
    pub fn new_with_vad(device_name: &str, vad_settings: VadSettings) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_input_device(&host, device_name)?;

        info!("Using audio device: {}", device.name()?);

        let _config = device.default_input_config()?;
        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(16000), // Whisper optimal
            buffer_size: cpal::BufferSize::Default,
        };
        let sample_rate = config.sample_rate.0;

        Ok(Self {
            device,
            config,
            samples: Arc::new(Mutex::new(Vec::new())),
            active_stream: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(RecordingState::Idle)),
            first_sample_seen: Arc::new(AtomicBool::new(false)),
            vad: VadProcessor::new(vad_settings, sample_rate),
        })
    }

    /// Start recording audio, properly managing stream lifecycle
    pub async fn start_recording(&self) -> Result<()> {
        bench_trace::event("audio_start_begin");
        let mut state = self.state.lock().unwrap();

        match *state {
            RecordingState::Recording => {
                return Err(anyhow::anyhow!("Recording already in progress"));
            }
            RecordingState::Stopping => {
                return Err(anyhow::anyhow!("Previous recording still stopping"));
            }
            RecordingState::Idle => {}
        }

        // Stop any existing stream before starting new one
        self.cleanup_stream();

        // Clear samples buffer for new recording
        {
            let mut samples = self.samples.lock().unwrap();
            samples.clear();
            samples.shrink_to_fit(); // Free memory from previous recordings
        }
        self.first_sample_seen.store(false, Ordering::Relaxed);

        debug!("Creating new audio stream");

        let samples_clone = self.samples.clone();
        let first_sample_seen = self.first_sample_seen.clone();
        let err_fn = |err| error!("Audio stream error: {}", err);

        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !data.is_empty() && !first_sample_seen.swap(true, Ordering::Relaxed) {
                    bench_trace::event_with_extra("audio_first_sample", || {
                        serde_json::json!({
                            "samples": data.len(),
                        })
                    });
                }
                if let Ok(mut samples) = samples_clone.lock() {
                    samples.extend_from_slice(data);
                }
            },
            err_fn,
            None,
        )?;

        stream.play()?;
        bench_trace::event("audio_stream_played");
        info!("Started audio recording");

        // Store stream for proper cleanup
        *self.active_stream.lock().unwrap() = Some(stream);
        *state = RecordingState::Recording;

        Ok(())
    }

    /// Stop recording and save audio to file
    pub async fn stop_recording(&self, output_path: PathBuf) -> Result<RecordedAudio> {
        bench_trace::event_with_extra("audio_stop_begin", || {
            serde_json::json!({
                "output_path": output_path.display().to_string(),
            })
        });
        let mut state = self.state.lock().unwrap();

        match *state {
            RecordingState::Idle => {
                return Err(anyhow::anyhow!("No recording in progress"));
            }
            RecordingState::Stopping => {
                return Err(anyhow::anyhow!("Recording already stopping"));
            }
            RecordingState::Recording => {}
        }

        *state = RecordingState::Stopping;
        drop(state); // Release lock before cleanup

        // Stop and cleanup stream
        self.cleanup_stream();

        // Move samples out without copying the full recording buffer.
        let samples = take_samples(&self.samples);
        bench_trace::event_with_extra("samples_taken", || {
            serde_json::json!({
                "samples": samples.len(),
                "sample_rate": self.config.sample_rate.0,
            })
        });

        if samples.is_empty() {
            *self.state.lock().unwrap() = RecordingState::Idle;
            bench_trace::event_with_extra("trial_error", || {
                serde_json::json!({
                    "phase": "samples_taken",
                    "error": "No audio samples recorded",
                })
            });
            return Err(anyhow::anyhow!("No audio samples recorded"));
        }

        info!("Stopping recording, {} samples captured", samples.len());

        let vad_output = self.vad.process(samples);
        trace_vad_output(&vad_output);
        if vad_output.skipped {
            *self.state.lock().unwrap() = RecordingState::Idle;
            info!("VAD detected no speech; skipping WAV write and transcription");
            return Ok(RecordedAudio::NoSpeech);
        }

        let write_result = write_samples_to_wav(&output_path, &vad_output.samples);

        *self.state.lock().unwrap() = RecordingState::Idle;

        write_result?;

        info!("Audio saved to: {:?}", output_path);
        Ok(RecordedAudio::Speech(output_path))
    }

    /// Cleanup any active stream
    fn cleanup_stream(&self) {
        let mut active_stream = self.active_stream.lock().unwrap();
        if let Some(stream) = active_stream.take() {
            debug!("Cleaning up audio stream");
            // Stream is automatically stopped when dropped
            drop(stream);
            bench_trace::event("audio_stream_dropped");
        }
    }
}

fn trace_vad_output(output: &VadOutput) {
    bench_trace::event_with_extra("vad_trim_done", || {
        serde_json::json!({
            "engine": output.engine,
            "input_samples": output.input_samples,
            "output_samples": output.output_samples,
            "skipped": output.skipped,
        })
    });
}

fn take_samples(samples: &Arc<Mutex<Vec<f32>>>) -> Vec<f32> {
    let mut samples = samples.lock().unwrap();
    std::mem::take(&mut *samples)
}

fn write_samples_to_wav(output_path: &Path, samples: &[f32]) -> Result<()> {
    bench_trace::event_with_extra("wav_write_begin", || {
        serde_json::json!({
            "output_path": output_path.display().to_string(),
            "samples": samples.len(),
        })
    });

    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(output_path, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    let wav_bytes = std::fs::metadata(output_path)
        .map(|metadata| metadata.len())
        .ok();
    bench_trace::event_with_extra("wav_write_done", || {
        serde_json::json!({
            "output_path": output_path.display().to_string(),
            "samples": samples.len(),
            "wav_bytes": wav_bytes,
        })
    });

    Ok(())
}

impl Drop for AudioStreamManager {
    fn drop(&mut self) {
        debug!("Dropping AudioStreamManager, cleaning up resources");
        self.cleanup_stream();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_ci() -> bool {
        std::env::var("CI").is_ok()
            || std::env::var("GITHUB_ACTIONS").is_ok()
            || std::env::var("GITLAB_CI").is_ok()
            || std::env::var("TRAVIS").is_ok()
    }

    #[tokio::test]
    async fn test_audio_stream_manager_creation() {
        if is_ci() {
            // Skip audio tests in CI - no audio devices available
            return;
        }

        // This test may fail in CI without audio devices
        let _manager = AudioStreamManager::new_with_vad("default", VadSettings::default());
    }

    #[test]
    fn take_samples_moves_recorded_samples_out_of_shared_buffer() {
        let samples = Arc::new(Mutex::new(vec![0.1, 0.2, 0.3]));

        let taken = take_samples(&samples);

        assert_eq!(taken, vec![0.1, 0.2, 0.3]);
        assert!(samples.lock().unwrap().is_empty());
    }
}
