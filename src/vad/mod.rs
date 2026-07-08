use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::{debug, info, warn};
use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

const DEFAULT_SAMPLE_RATE: u32 = 16_000;
// Filename as produced by whisper.cpp's models/download-vad-model.sh
const DEFAULT_VAD_MODEL: &str = ".local/share/chezwizper/whisper/models/ggml-silero-v5.1.2.bin";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VadEngine {
    Auto,
    Silero,
    Amplitude,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VadSettings {
    pub enabled: bool,
    pub engine: VadEngine,
    pub threshold: f32,
    pub min_speech_ms: u32,
    pub pad_ms: u32,
    pub model_path: Option<PathBuf>,
}

impl Default for VadSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: VadEngine::Auto,
            threshold: 0.02,
            min_speech_ms: 100,
            pad_ms: 200,
            model_path: None,
        }
    }
}

#[derive(Debug)]
pub struct VadOutput {
    pub samples: Vec<f32>,
    pub input_samples: usize,
    pub output_samples: usize,
    pub skipped: bool,
    pub engine: &'static str,
}

pub struct VadProcessor {
    settings: VadSettings,
    sample_rate: u32,
    engine: ActiveVadEngine,
}

enum ActiveVadEngine {
    Disabled,
    Amplitude,
    Silero(Mutex<WhisperVadContext>),
}

impl VadProcessor {
    pub fn new(settings: VadSettings, sample_rate: u32) -> Self {
        let sample_rate = sample_rate.max(1);
        let engine = if settings.enabled {
            Self::select_engine(&settings)
        } else {
            ActiveVadEngine::Disabled
        };

        Self {
            settings,
            sample_rate,
            engine,
        }
    }

    pub fn process(&self, samples: Vec<f32>) -> VadOutput {
        let input_samples = samples.len();
        let (samples, skipped, engine) = match &self.engine {
            ActiveVadEngine::Disabled => (samples, false, "disabled"),
            ActiveVadEngine::Amplitude => {
                let result = trim_with_amplitude(&samples, &self.settings, self.sample_rate);
                (result.samples, result.skipped, "amplitude")
            }
            ActiveVadEngine::Silero(context) => {
                match trim_with_silero(&samples, &self.settings, self.sample_rate, context) {
                    Ok(result) => (result.samples, result.skipped, "silero"),
                    Err(error) => {
                        warn!("Silero VAD failed, falling back to amplitude VAD: {error}");
                        let result =
                            trim_with_amplitude(&samples, &self.settings, self.sample_rate);
                        (result.samples, result.skipped, "amplitude")
                    }
                }
            }
        };

        let output_samples = samples.len();
        VadOutput {
            samples,
            input_samples,
            output_samples,
            skipped,
            engine,
        }
    }

    fn select_engine(settings: &VadSettings) -> ActiveVadEngine {
        match settings.engine {
            VadEngine::Amplitude => ActiveVadEngine::Amplitude,
            VadEngine::Auto | VadEngine::Silero => {
                if let Some(model_path) = resolve_silero_model_path(settings) {
                    match load_silero(&model_path) {
                        Ok(context) => {
                            info!("Using Silero VAD model at {}", model_path.display());
                            ActiveVadEngine::Silero(Mutex::new(context))
                        }
                        Err(error) => {
                            warn!(
                                "Could not load Silero VAD model at {}: {error}; using amplitude VAD",
                                model_path.display()
                            );
                            ActiveVadEngine::Amplitude
                        }
                    }
                } else {
                    if matches!(settings.engine, VadEngine::Silero) {
                        warn!("Silero VAD requested but no model was found; using amplitude VAD");
                    }
                    ActiveVadEngine::Amplitude
                }
            }
        }
    }
}

fn load_silero(
    model_path: &std::path::Path,
) -> Result<WhisperVadContext, whisper_rs::WhisperError> {
    let mut params = WhisperVadContextParams::default();
    params.set_n_threads(1);
    params.set_use_gpu(false);
    WhisperVadContext::new(&model_path.display().to_string(), params)
}

fn resolve_silero_model_path(settings: &VadSettings) -> Option<PathBuf> {
    if let Some(path) = &settings.model_path {
        return path.exists().then(|| path.clone());
    }

    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(DEFAULT_VAD_MODEL))
        .filter(|path| path.exists())
}

struct TrimResult {
    samples: Vec<f32>,
    skipped: bool,
}

fn trim_with_silero(
    samples: &[f32],
    settings: &VadSettings,
    sample_rate: u32,
    context: &Mutex<WhisperVadContext>,
) -> anyhow::Result<TrimResult> {
    let mut params = WhisperVadParams::new();
    params.set_threshold(settings.threshold.clamp(0.0, 1.0));
    params.set_min_speech_duration(settings.min_speech_ms as i32);
    params.set_speech_pad(settings.pad_ms as i32);

    let mut context = context.lock().expect("Silero VAD mutex poisoned");
    let mut start = samples.len();
    let mut end = 0;

    for segment in context.segments_from_samples(params, samples)? {
        let segment_start = timestamp_to_sample(segment.start, sample_rate);
        let segment_end = timestamp_to_sample(segment.end, sample_rate);
        start = start.min(segment_start);
        end = end.max(segment_end);
    }

    Ok(trim_range(samples, start, end))
}

fn trim_with_amplitude(samples: &[f32], settings: &VadSettings, sample_rate: u32) -> TrimResult {
    let window = samples_for_ms(sample_rate, 30).max(1) as usize;
    let hop = samples_for_ms(sample_rate, 10).max(1) as usize;
    let min_speech_samples = samples_for_ms(sample_rate, settings.min_speech_ms).max(hop as u32);
    let pad = samples_for_ms(sample_rate, settings.pad_ms) as usize;

    let mut first = None;
    let mut last = None;
    let mut run_start = None;
    let mut run_end = 0usize;

    let mut offset = 0usize;
    while offset < samples.len() {
        let end = (offset + window).min(samples.len());
        if window_is_speech(&samples[offset..end], settings.threshold) {
            if run_start.is_none() {
                run_start = Some(offset);
            }
            run_end = end;
        } else if let Some(start) = run_start.take() {
            if run_end.saturating_sub(start) >= min_speech_samples as usize {
                first.get_or_insert(start);
                last = Some(run_end);
            }
        }

        offset += hop;
    }

    if let Some(start) = run_start {
        if run_end.saturating_sub(start) >= min_speech_samples as usize {
            first.get_or_insert(start);
            last = Some(run_end);
        }
    }

    if let (Some(start), Some(end)) = (first, last) {
        let padded_start = start.saturating_sub(pad);
        let padded_end = (end + pad).min(samples.len());
        debug!(
            "Amplitude VAD trimmed {} samples to {}..{}",
            samples.len(),
            padded_start,
            padded_end
        );
        TrimResult {
            samples: samples[padded_start..padded_end].to_vec(),
            skipped: false,
        }
    } else {
        TrimResult {
            samples: Vec::new(),
            skipped: true,
        }
    }
}

fn trim_range(samples: &[f32], start: usize, end: usize) -> TrimResult {
    if start >= end || start >= samples.len() {
        return TrimResult {
            samples: Vec::new(),
            skipped: true,
        };
    }

    let end = end.min(samples.len());
    TrimResult {
        samples: samples[start..end].to_vec(),
        skipped: false,
    }
}

fn timestamp_to_sample(centiseconds: f32, sample_rate: u32) -> usize {
    ((centiseconds / 100.0) * sample_rate as f32).round() as usize
}

fn samples_for_ms(sample_rate: u32, ms: u32) -> u32 {
    sample_rate.saturating_mul(ms) / 1_000
}

fn window_is_speech(samples: &[f32], threshold: f32) -> bool {
    if samples.is_empty() {
        return false;
    }

    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();

    peak >= threshold || rms >= threshold
}

impl Default for VadProcessor {
    fn default() -> Self {
        Self::new(VadSettings::default(), DEFAULT_SAMPLE_RATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VadSettings {
        VadSettings {
            enabled: true,
            engine: VadEngine::Amplitude,
            threshold: 0.02,
            min_speech_ms: 100,
            pad_ms: 200,
            model_path: None,
        }
    }

    #[test]
    fn amplitude_gate_skips_silence() {
        let samples = vec![0.0; 16_000];

        let result = VadProcessor::new(cfg(), 16_000).process(samples);

        assert!(result.skipped);
        assert!(result.samples.is_empty());
    }

    #[test]
    fn amplitude_gate_trims_with_padding() {
        let mut samples = vec![0.0; 16_000];
        samples.extend(vec![0.08; 8_000]);
        samples.extend(vec![0.0; 16_000]);

        let result = VadProcessor::new(cfg(), 16_000).process(samples);

        assert!(!result.skipped);
        assert!(result.samples.len() >= 14_400);
        assert!(result.samples.len() <= 15_360);
    }

    #[test]
    fn amplitude_gate_skips_audio_below_threshold() {
        let samples = vec![0.005; 16_000];

        let result = VadProcessor::new(cfg(), 16_000).process(samples);

        assert!(result.skipped);
        assert!(result.samples.is_empty());
    }
}
