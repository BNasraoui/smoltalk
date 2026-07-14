#![allow(clippy::arc_with_non_send_sync)]

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use std::path::Path;
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

struct RecordingStateResetGuard {
    state: Arc<Mutex<RecordingState>>,
}

impl RecordingStateResetGuard {
    fn new(state: Arc<Mutex<RecordingState>>) -> Self {
        Self { state }
    }
}

impl Drop for RecordingStateResetGuard {
    fn drop(&mut self) {
        *self.state.lock().unwrap() = RecordingState::Idle;
    }
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
    Speech(Vec<f32>),
    NoSpeech,
}

#[derive(Debug)]
pub struct FinalRecordingSnapshot {
    pub samples: Vec<f32>,
    pub speech_end: usize,
}

#[derive(Clone)]
pub struct RecordingBuffer {
    samples: Arc<Mutex<Vec<f32>>>,
}

impl RecordingBuffer {
    pub(crate) fn new(samples: Arc<Mutex<Vec<f32>>>) -> Self {
        Self { samples }
    }

    pub fn read_from(&self, offset: usize) -> Vec<f32> {
        let samples = self.samples.lock().unwrap();
        samples.get(offset..).unwrap_or_default().to_vec()
    }
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

    pub fn recording_buffer(&self) -> RecordingBuffer {
        RecordingBuffer::new(self.samples.clone())
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate.0
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

    /// Stop recording and return the VAD-trimmed audio samples.
    pub async fn stop_recording(&self) -> Result<RecordedAudio> {
        self.stop_recording_inner(false)
            .await
            .map(|(audio, _)| audio)
    }

    pub async fn stop_recording_with_snapshot(
        &self,
    ) -> Result<(RecordedAudio, FinalRecordingSnapshot)> {
        let (audio, snapshot) = self.stop_recording_inner(true).await?;
        Ok((
            audio,
            snapshot.context("final recording snapshot was not retained")?,
        ))
    }

    pub fn cancel_recording(&self) {
        bench_trace::event("audio_cancel_begin");
        discard_recording(&self.active_stream, &self.state, &self.samples);
        bench_trace::event("audio_cancel_done");
    }

    async fn stop_recording_inner(
        &self,
        retain_snapshot: bool,
    ) -> Result<(RecordedAudio, Option<FinalRecordingSnapshot>)> {
        bench_trace::event("audio_stop_begin");
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
        let _state_reset_guard = RecordingStateResetGuard::new(self.state.clone());

        // Stop and cleanup stream
        self.cleanup_stream();

        // Move samples out without copying the full recording buffer.
        let (samples, raw_snapshot) = take_samples_with_snapshot(&self.samples, retain_snapshot);
        bench_trace::event_with_extra("samples_taken", || {
            serde_json::json!({
                "samples": samples.len(),
                "sample_rate": self.config.sample_rate.0,
            })
        });

        if samples.is_empty() {
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
        let snapshot = raw_snapshot.map(|samples| FinalRecordingSnapshot {
            samples,
            speech_end: vad_output
                .speech_range
                .as_ref()
                .map_or(0, |range| range.end),
        });
        if vad_output.skipped {
            info!("VAD detected no speech; skipping transcription");
        }

        Ok((recorded_audio_from_vad(vad_output), snapshot))
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

fn discard_recording(
    active_stream: &Arc<Mutex<Option<cpal::Stream>>>,
    state: &Arc<Mutex<RecordingState>>,
    samples: &Arc<Mutex<Vec<f32>>>,
) {
    active_stream.lock().unwrap().take();
    samples.lock().unwrap().clear();
    *state.lock().unwrap() = RecordingState::Idle;
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

fn recorded_audio_from_vad(output: VadOutput) -> RecordedAudio {
    if output.skipped {
        RecordedAudio::NoSpeech
    } else {
        RecordedAudio::Speech(output.samples)
    }
}

fn take_samples_with_snapshot(
    samples: &Arc<Mutex<Vec<f32>>>,
    retain_snapshot: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut samples = samples.lock().unwrap();
    let samples = std::mem::take(&mut *samples);
    let snapshot = retain_snapshot.then(|| samples.clone());
    (samples, snapshot)
}

pub(crate) fn write_samples_to_wav(output_path: &Path, samples: &[f32]) -> Result<()> {
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

        let (taken, snapshot) = take_samples_with_snapshot(&samples, false);

        assert_eq!(taken, vec![0.1, 0.2, 0.3]);
        assert_eq!(snapshot, None);
        assert!(samples.lock().unwrap().is_empty());
    }

    #[test]
    fn recording_buffer_reads_new_samples_without_consuming_them() {
        let samples = Arc::new(Mutex::new(vec![0.1, 0.2, 0.3, 0.4]));
        let buffer = RecordingBuffer {
            samples: samples.clone(),
        };

        assert_eq!(buffer.read_from(2), vec![0.3, 0.4]);
        assert_eq!(*samples.lock().unwrap(), vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn taking_samples_can_retain_a_final_chunking_snapshot() {
        let samples = Arc::new(Mutex::new(vec![0.1, 0.2, 0.3]));

        let (taken, snapshot) = take_samples_with_snapshot(&samples, true);

        assert_eq!(taken, vec![0.1, 0.2, 0.3]);
        assert_eq!(snapshot, Some(vec![0.1, 0.2, 0.3]));
        assert!(samples.lock().unwrap().is_empty());
    }

    #[test]
    fn vad_speech_result_owns_trimmed_samples() {
        let trimmed = vec![0.25, -0.5, 0.75];
        let output = VadOutput {
            samples: trimmed.clone(),
            speech_range: Some(4..7),
            input_samples: 11,
            output_samples: trimmed.len(),
            skipped: false,
            engine: "test",
        };

        match recorded_audio_from_vad(output) {
            RecordedAudio::Speech(samples) => assert_eq!(samples, trimmed),
            RecordedAudio::NoSpeech => panic!("speech samples should not be discarded"),
        }
    }

    #[test]
    fn skipped_vad_result_returns_no_speech_without_creating_a_wav() {
        let directory = tempfile::tempdir().unwrap();
        let unexpected_wav = directory.path().join("unexpected.wav");
        let output = VadOutput {
            samples: Vec::new(),
            speech_range: None,
            input_samples: 100,
            output_samples: 0,
            skipped: true,
            engine: "test",
        };

        assert!(matches!(
            recorded_audio_from_vad(output),
            RecordedAudio::NoSpeech
        ));
        assert!(!unexpected_wav.exists());
    }

    #[test]
    fn recording_state_reset_guard_restores_idle_during_unwind() {
        let state = Arc::new(Mutex::new(RecordingState::Stopping));
        let panic_state = state.clone();

        let result = std::panic::catch_unwind(move || {
            let _guard = RecordingStateResetGuard::new(panic_state);
            panic!("injected stop panic");
        });

        assert!(result.is_err());
        assert_eq!(*state.lock().unwrap(), RecordingState::Idle);
    }

    #[test]
    fn discard_recording_drops_samples_and_restores_idle() {
        let active_stream = Arc::new(Mutex::new(None));
        let state = Arc::new(Mutex::new(RecordingState::Recording));
        let samples = Arc::new(Mutex::new(vec![0.1, 0.2, 0.3]));

        discard_recording(&active_stream, &state, &samples);

        assert_eq!(*state.lock().unwrap(), RecordingState::Idle);
        assert!(samples.lock().unwrap().is_empty());
        assert!(active_stream.lock().unwrap().is_none());
    }
}
