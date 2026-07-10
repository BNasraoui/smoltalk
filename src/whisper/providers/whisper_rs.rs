use anyhow::{Context, Result};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::bench_trace;
use crate::whisper::provider::{AudioFileRetention, ModelStatusSnapshot, TranscriptionProvider};
use crate::whisper::AudioCtxConfig;

const WHISPER_SAMPLE_RATE: usize = 16_000;
const AUDIO_CTX_FULL: u32 = 1500;
// Contexts below ~640 destabilize decoding into hallucination/repetition loops
// (measured 2026-07-08 on a 30-phrase real-speech corpus: ac=512 blew up on 3/30
// files, ac=640 matched full-window WER with 2.6x faster encode).
const AUDIO_CTX_MIN: u32 = 640;
const AUDIO_CTX_SAFETY_MARGIN: u32 = 128;

pub struct WhisperRsProvider {
    model_path: PathBuf,
    options: WhisperRsOptions,
    inner: Arc<Mutex<WhisperRsInner>>,
}

#[derive(Clone, Debug)]
pub(crate) struct WhisperRsOptions {
    pub threads: Option<u32>,
    pub beam_size: Option<u32>,
    pub best_of: Option<u32>,
    pub no_fallback: Option<bool>,
    pub keep_warm_for_secs: Option<u64>,
    pub initial_prompt: Option<String>,
    pub audio_ctx: AudioCtxConfig,
}

impl Default for WhisperRsOptions {
    fn default() -> Self {
        Self {
            threads: None,
            beam_size: None,
            best_of: None,
            no_fallback: None,
            keep_warm_for_secs: Some(0),
            initial_prompt: None,
            audio_ctx: AudioCtxConfig::Auto,
        }
    }
}

struct WhisperRsInner {
    context: Option<WhisperContext>,
    whisper_state: Option<WhisperState>,
    state: WarmModelState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WhisperRsModelStatus {
    Cold,
    Loading,
    Warm,
    IdleUnloaded,
    Error,
}

impl WhisperRsModelStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Loading => "loading",
            Self::Warm => "warm",
            Self::IdleUnloaded => "idle-unloaded",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnloadReason {
    Manual,
    IdleTimer,
}

#[derive(Debug)]
struct WarmModelState {
    status: WhisperRsModelStatus,
    keep_warm_for: Option<Duration>,
    last_used_at: Option<Instant>,
    error: Option<String>,
}

impl WarmModelState {
    fn new(keep_warm_for: Option<Duration>) -> Self {
        Self {
            status: WhisperRsModelStatus::Cold,
            keep_warm_for,
            last_used_at: None,
            error: None,
        }
    }

    fn status(&self) -> WhisperRsModelStatus {
        self.status
    }

    fn error_message(&self) -> Option<&str> {
        self.error.as_deref()
    }

    fn mark_loading(&mut self) {
        self.status = WhisperRsModelStatus::Loading;
        self.error = None;
        trace_state_transition(self.status, None);
    }

    fn mark_warm(&mut self) {
        self.mark_warm_at(Instant::now());
    }

    fn mark_warm_at(&mut self, now: Instant) {
        self.status = WhisperRsModelStatus::Warm;
        self.last_used_at = Some(now);
        self.error = None;
        trace_state_transition(self.status, None);
    }

    fn mark_unloaded(&mut self, reason: UnloadReason) {
        self.status = match reason {
            UnloadReason::Manual => WhisperRsModelStatus::Cold,
            UnloadReason::IdleTimer => WhisperRsModelStatus::IdleUnloaded,
        };
        self.last_used_at = None;
        self.error = None;
        trace_state_transition(
            self.status,
            Some(match reason {
                UnloadReason::Manual => "manual",
                UnloadReason::IdleTimer => "idle_timer",
            }),
        );
    }

    fn mark_error(&mut self, error: impl Into<String>) {
        let error = error.into();
        self.status = WhisperRsModelStatus::Error;
        self.error = Some(error.clone());
        trace_state_transition(self.status, Some(&error));
    }

    fn should_idle_unload(&self, now: Instant) -> bool {
        let Some(keep_warm_for) = self.keep_warm_for else {
            return false;
        };
        if keep_warm_for.is_zero() {
            return false;
        }
        let Some(last_used_at) = self.last_used_at else {
            return false;
        };

        self.status == WhisperRsModelStatus::Warm
            && now.duration_since(last_used_at) >= keep_warm_for
    }
}

impl WhisperRsProvider {
    pub(crate) fn new(
        model: String,
        model_path: Option<String>,
        options: WhisperRsOptions,
    ) -> Result<Self> {
        let model_path = resolve_model_path(model, model_path)?;
        let keep_warm_for = options.keep_warm_for_secs.map(Duration::from_secs);

        Ok(Self {
            model_path,
            options,
            inner: Arc::new(Mutex::new(WhisperRsInner {
                context: None,
                whisper_state: None,
                state: WarmModelState::new(keep_warm_for),
            })),
        })
    }

    fn ensure_loaded(&self) -> Result<()> {
        self.unload_if_idle_expired();

        {
            let inner = self.inner.lock().unwrap();
            if inner.context.is_some() && inner.whisper_state.is_some() {
                return Ok(());
            }
        }

        {
            let mut inner = self.inner.lock().unwrap();
            inner.state.mark_loading();
        }

        bench_trace::event_with_extra("model_load_begin", || {
            serde_json::json!({
                "provider": "whisper-rs",
                "model_path": self.model_path.display().to_string(),
            })
        });

        let model_path = self
            .model_path
            .to_str()
            .context("whisper-rs model path must be valid UTF-8")?;
        let started_at = Instant::now();
        let mut context_params = WhisperContextParameters::default();
        // Measured ~26% slower on CPU-only inference (whisper-cli -fa, 2026-07-08);
        // flash attention only pays off on GPU backends.
        context_params.flash_attn(false);

        let result = WhisperContext::new_with_params(model_path, context_params)
            .with_context(|| {
                format!(
                    "failed to load whisper-rs model {}",
                    self.model_path.display()
                )
            })
            .and_then(|context| {
                let state = context
                    .create_state()
                    .context("failed to create whisper-rs state")?;
                Ok((context, state))
            });

        bench_trace::event_with_extra("model_load_end", || {
            serde_json::json!({
                "provider": "whisper-rs",
                "model_path": self.model_path.display().to_string(),
                "elapsed_ms": started_at.elapsed().as_millis(),
                "success": result.is_ok(),
            })
        });

        let mut inner = self.inner.lock().unwrap();
        match result {
            Ok((context, whisper_state)) => {
                inner.context = Some(context);
                inner.whisper_state = Some(whisper_state);
                inner.state.mark_warm();
                Ok(())
            }
            Err(error) => {
                inner.whisper_state = None;
                inner.context = None;
                inner.state.mark_error(error.to_string());
                Err(error)
            }
        }
    }

    fn unload_if_idle_expired(&self) {
        let mut inner = self.inner.lock().unwrap();
        if inner.context.is_some() && inner.state.should_idle_unload(Instant::now()) {
            inner.whisper_state = None;
            inner.context = None;
            inner.state.mark_unloaded(UnloadReason::IdleTimer);
        }
    }

    fn transcribe_loaded_samples(&self, samples: &[f32], language: &str) -> Result<String> {
        let mut inner = self.inner.lock().unwrap();
        let text = {
            let params = self.full_params(language, samples.len());
            let state = inner
                .whisper_state
                .as_mut()
                .context("whisper-rs state is not available")?;

            state
                .full(params, samples)
                .context("whisper-rs transcription failed")?;

            let mut text = String::new();
            for segment in state.as_iter() {
                let segment_text = segment.to_str_lossy()?;
                text.push_str(segment_text.as_ref());
            }

            text
        };

        inner.state.mark_warm();
        Ok(text.trim().to_string())
    }

    fn full_params<'a>(&'a self, language: &'a str, sample_count: usize) -> FullParams<'a, 'a> {
        let strategy = if let Some(beam_size) = self.options.beam_size {
            SamplingStrategy::BeamSearch {
                beam_size: beam_size.max(1) as i32,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy {
                best_of: self.options.best_of.unwrap_or(1).max(1) as i32,
            }
        };

        let mut params = FullParams::new(strategy);
        params.set_language(Some(language));
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);
        params.set_no_context(true);

        if let Some(threads) = self.options.threads {
            params.set_n_threads(threads.max(1) as i32);
        }

        if self.options.no_fallback.unwrap_or(false) {
            params.set_temperature_inc(0.0);
        }

        if let Some(initial_prompt) = &self.options.initial_prompt {
            params.set_initial_prompt(initial_prompt);
        }

        let audio_ctx = match self.options.audio_ctx {
            AudioCtxConfig::Auto => audio_ctx_for_sample_count(sample_count),
            AudioCtxConfig::Fixed(value) => value.clamp(AUDIO_CTX_MIN, AUDIO_CTX_FULL),
            AudioCtxConfig::Full => 0,
        };
        params.set_audio_ctx(audio_ctx as i32);

        params
    }
}

impl TranscriptionProvider for WhisperRsProvider {
    fn name(&self) -> &'static str {
        "whisper-rs"
    }

    fn is_available(&self) -> bool {
        self.model_path.exists()
    }

    fn supports_chunking(&self) -> bool {
        true
    }

    fn prepare<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move { self.ensure_loaded() })
    }

    fn model_status(&self) -> Option<ModelStatusSnapshot> {
        self.unload_if_idle_expired();
        let inner = self.inner.lock().unwrap();

        Some(ModelStatusSnapshot {
            provider: self.name(),
            state: inner.state.status().as_str(),
            error: inner.state.error_message().map(str::to_string),
        })
    }

    fn unload_model(&self) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.whisper_state = None;
        inner.context = None;
        inner.state.mark_unloaded(UnloadReason::Manual);
        Ok(())
    }

    fn recording_complete(&self) -> Result<()> {
        if self.options.keep_warm_for_secs == Some(0) {
            self.unload_model()?;
        }
        Ok(())
    }

    fn reload_model<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.unload_model()?;
            self.ensure_loaded()
        })
    }

    fn transcribe<'a>(
        &'a self,
        audio_path: &'a Path,
        language: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            self.ensure_loaded()?;
            let samples = read_wav_samples(audio_path)?;
            self.transcribe_loaded_samples(&samples, language)
        })
    }

    fn transcribe_samples<'a>(
        &'a self,
        samples: &'a [f32],
        language: &'a str,
        _retention: AudioFileRetention,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            self.ensure_loaded()?;
            self.transcribe_loaded_samples(samples, language)
        })
    }
}

fn resolve_model_path(model: String, model_path: Option<String>) -> Result<PathBuf> {
    if let Some(model_path) = model_path {
        return Ok(PathBuf::from(model_path));
    }

    let models_dir = dirs::data_dir()
        .context("failed to determine data directory")?
        .join("chezwizper")
        .join("whisper")
        .join("models");

    if model == "base" || model == "base.en" {
        for candidate in [
            models_dir.join("ggml-base.en-q5_0.bin"),
            models_dir.join("ggml-base.en.bin"),
            models_dir.join("ggml-base.bin"),
        ] {
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        return Ok(models_dir.join("ggml-base.en-q5_0.bin"));
    }

    Ok(models_dir.join(format!("ggml-{model}.bin")))
}

fn read_wav_samples(audio_path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(audio_path)
        .with_context(|| format!("failed to open WAV {}", audio_path.display()))?;
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(anyhow::anyhow!(
            "whisper-rs provider expects mono WAV, got {} channels",
            spec.channels
        ));
    }

    if spec.sample_rate != 16_000 {
        return Err(anyhow::anyhow!(
            "whisper-rs provider expects 16 kHz WAV, got {} Hz",
            spec.sample_rate
        ));
    }

    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read f32 WAV samples"),
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|sample| f32::from(sample) / f32::from(i16::MAX)))
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read i16 WAV samples"),
        _ => Err(anyhow::anyhow!(
            "unsupported WAV format for whisper-rs: {:?} {} bits",
            spec.sample_format,
            spec.bits_per_sample
        )),
    }
}

pub(crate) fn audio_ctx_for_sample_count(sample_count: usize) -> u32 {
    let frames = sample_count.div_ceil(WHISPER_SAMPLE_RATE / 50) as u32;
    frames
        .saturating_add(AUDIO_CTX_SAFETY_MARGIN)
        .clamp(AUDIO_CTX_MIN, AUDIO_CTX_FULL)
}

fn trace_state_transition(status: WhisperRsModelStatus, reason: Option<&str>) {
    bench_trace::event_with_extra("model_state_transition", || {
        serde_json::json!({
            "provider": "whisper-rs",
            "state": status.as_str(),
            "reason": reason,
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper::provider::AudioFileRetention;
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    fn temporary_chezwizper_wavs() -> HashSet<PathBuf> {
        std::fs::read_dir(std::env::temp_dir())
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("chezwizper-") && name.ends_with(".wav"))
            })
            .collect()
    }

    #[test]
    fn state_machine_tracks_load_success_and_manual_unload() {
        let mut state = WarmModelState::new(Some(Duration::from_secs(30)));

        assert_eq!(state.status(), WhisperRsModelStatus::Cold);

        state.mark_loading();
        assert_eq!(state.status(), WhisperRsModelStatus::Loading);

        state.mark_warm();
        assert_eq!(state.status(), WhisperRsModelStatus::Warm);

        state.mark_unloaded(UnloadReason::Manual);
        assert_eq!(state.status(), WhisperRsModelStatus::Cold);
    }

    #[test]
    fn state_machine_reports_idle_unloaded_after_idle_timer() {
        let mut state = WarmModelState::new(Some(Duration::from_secs(5)));
        state.mark_warm();

        state.mark_unloaded(UnloadReason::IdleTimer);

        assert_eq!(state.status(), WhisperRsModelStatus::IdleUnloaded);
    }

    #[test]
    fn state_machine_reports_error_until_next_load_attempt() {
        let mut state = WarmModelState::new(None);

        state.mark_error("model load failed");
        assert_eq!(state.status(), WhisperRsModelStatus::Error);
        assert_eq!(state.error_message(), Some("model load failed"));

        state.mark_loading();
        assert_eq!(state.status(), WhisperRsModelStatus::Loading);
        assert_eq!(state.error_message(), None);
    }

    #[test]
    fn idle_timer_only_expires_after_configured_warm_duration() {
        let mut state = WarmModelState::new(Some(Duration::from_secs(10)));
        let now = Instant::now();

        state.mark_warm_at(now);

        assert!(!state.should_idle_unload(now + Duration::from_secs(9)));
        assert!(state.should_idle_unload(now + Duration::from_secs(10)));
    }

    #[test]
    fn zero_duration_waits_for_recording_completion_to_unload() {
        let mut state = WarmModelState::new(Some(Duration::ZERO));
        let now = Instant::now();

        state.mark_warm_at(now);

        assert!(!state.should_idle_unload(now));
    }

    #[test]
    fn disabled_idle_timer_never_expires() {
        let mut state = WarmModelState::new(None);
        let now = Instant::now();

        state.mark_warm_at(now);

        assert!(!state.should_idle_unload(now + Duration::from_secs(3600)));
    }

    #[test]
    fn options_default_to_releasing_the_model_between_recordings() {
        assert_eq!(WhisperRsOptions::default().keep_warm_for_secs, Some(0));
    }

    #[test]
    fn recording_completion_releases_the_default_cold_provider() {
        let provider = WhisperRsProvider::new(
            "base.en".to_string(),
            Some("/tmp/missing-whisper-model.bin".to_string()),
            WhisperRsOptions::default(),
        )
        .unwrap();
        provider.inner.lock().unwrap().state.mark_warm();

        provider.recording_complete().unwrap();

        assert_eq!(
            provider.inner.lock().unwrap().state.status(),
            WhisperRsModelStatus::Cold
        );
    }

    #[test]
    fn recording_completion_retains_an_explicitly_warm_provider() {
        let provider = WhisperRsProvider::new(
            "base.en".to_string(),
            Some("/tmp/missing-whisper-model.bin".to_string()),
            WhisperRsOptions {
                keep_warm_for_secs: Some(300),
                ..Default::default()
            },
        )
        .unwrap();
        provider.inner.lock().unwrap().state.mark_warm();

        provider.recording_complete().unwrap();

        assert_eq!(
            provider.inner.lock().unwrap().state.status(),
            WhisperRsModelStatus::Warm
        );
    }

    #[test]
    fn audio_ctx_for_short_clip_uses_stability_floor() {
        assert_eq!(audio_ctx_for_sample_count(WHISPER_SAMPLE_RATE), 640);
    }

    #[test]
    fn audio_ctx_for_exactly_30_seconds_clamps_to_full_context() {
        assert_eq!(audio_ctx_for_sample_count(WHISPER_SAMPLE_RATE * 30), 1500);
    }

    #[test]
    fn audio_ctx_for_long_clip_clamps_to_full_context() {
        assert_eq!(audio_ctx_for_sample_count(WHISPER_SAMPLE_RATE * 45), 1500);
    }

    #[test]
    fn audio_ctx_applies_safety_margin_above_raw_frame_count() {
        // 15s -> 750 raw frames + 128 margin, above the 640 floor
        assert_eq!(audio_ctx_for_sample_count(WHISPER_SAMPLE_RATE * 15), 878);
    }

    #[test]
    fn provider_supports_pause_chunking() {
        let provider = WhisperRsProvider::new(
            "base.en".to_string(),
            Some("/tmp/missing-whisper-model.bin".to_string()),
            WhisperRsOptions::default(),
        )
        .unwrap();

        assert!(provider.supports_chunking());
    }

    #[test]
    fn path_input_reads_f32_wav_samples() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("float.wav");
        let expected = vec![0.25, -0.5, 0.75];
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for sample in &expected {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();

        assert_eq!(read_wav_samples(&path).unwrap(), expected);
    }

    #[test]
    fn path_input_reads_i16_wav_samples() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("integer.wav");
        let encoded = [i16::MIN + 1, 0, i16::MAX];
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for sample in encoded {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        assert_eq!(
            read_wav_samples(&path).unwrap(),
            encoded
                .into_iter()
                .map(|sample| f32::from(sample) / f32::from(i16::MAX))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn sample_input_does_not_materialize_a_wav_when_model_load_fails() {
        let provider = WhisperRsProvider::new(
            "base.en".to_string(),
            Some("/tmp/definitely-missing-chezwizper-model.bin".to_string()),
            WhisperRsOptions::default(),
        )
        .unwrap();
        let before = temporary_chezwizper_wavs();

        let result = provider
            .transcribe_samples(&[0.123_456, -0.654_321], "en", AudioFileRetention::Keep)
            .await;

        let created = temporary_chezwizper_wavs()
            .difference(&before)
            .cloned()
            .collect::<Vec<_>>();
        for path in &created {
            let _ = std::fs::remove_file(path);
        }
        assert!(result.is_err());
        assert!(created.is_empty(), "sample input created WAVs: {created:?}");
    }

    #[tokio::test]
    async fn model_backed_sample_input_transcribes_without_creating_a_wav() {
        let Some(data_dir) = dirs::data_dir() else {
            return;
        };
        let model_path = data_dir.join("chezwizper/whisper/models/ggml-base.en-q5_0.bin");
        let corpus_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bench-artifacts/corpus/wav/short_status.wav");
        if !model_path.exists() || !corpus_path.exists() {
            return;
        }

        let samples = read_wav_samples(&corpus_path).unwrap();
        let provider = WhisperRsProvider::new(
            "base.en".to_string(),
            Some(model_path.display().to_string()),
            WhisperRsOptions::default(),
        )
        .unwrap();
        let before = temporary_chezwizper_wavs();

        let text = provider
            .transcribe_samples(&samples, "en", AudioFileRetention::Keep)
            .await
            .unwrap();

        let created = temporary_chezwizper_wavs()
            .difference(&before)
            .cloned()
            .collect::<Vec<_>>();
        for path in &created {
            let _ = std::fs::remove_file(path);
        }
        assert!(!text.trim().is_empty());
        assert!(created.is_empty(), "sample input created WAVs: {created:?}");
    }
}
