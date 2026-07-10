use anyhow::{Context, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::PathBuf;
use tracing::{info, warn};

use crate::bench_trace;

pub mod provider;
mod providers;

use provider::{AudioFileRetention, ModelStatusSnapshot, TranscriptionProvider};
use providers::{
    OpenAIProvider, OpenAIWhisperCliProvider, WhisperCppOptions, WhisperCppProvider,
    WhisperRsOptions, WhisperRsProvider,
};

pub struct WhisperTranscriber {
    provider: Box<dyn TranscriptionProvider>,
    language: String,
}

impl WhisperTranscriber {
    pub fn auto_detect(config: ProviderConfig) -> Result<Self> {
        let language = config.language.clone().unwrap_or_else(|| "en".to_string());
        let provider = Self::auto_detect_provider(config)?;

        Ok(Self::from_provider(provider, language))
    }

    pub fn with_provider(provider_name: &str, config: ProviderConfig) -> Result<Self> {
        let language = config.language.clone().unwrap_or_else(|| "en".to_string());

        let provider: Box<dyn TranscriptionProvider> = match provider_name {
            "openai-api" => {
                let api_key = config
                    .api_key
                    .context("api_key is required for OpenAI API provider")?;

                let model = config.model.unwrap_or_else(|| "whisper-1".to_string());
                Box::new(OpenAIProvider::new(api_key, config.api_endpoint, model)?)
            }
            "openai-cli" => {
                let model = config.model.unwrap_or_else(|| "base".to_string());
                Box::new(OpenAIWhisperCliProvider::new(config.command_path, model)?)
            }
            "whisper-cpp" => {
                let options = config.whisper_cpp_options();
                let model = config.model.unwrap_or_else(|| "base".to_string());
                Box::new(WhisperCppProvider::new(
                    config.command_path,
                    model,
                    config.model_path,
                    options,
                )?)
            }
            "whisper-rs" => {
                let options = config.whisper_rs_options();
                let model = config.model.unwrap_or_else(|| "base.en".to_string());
                Box::new(WhisperRsProvider::new(model, config.model_path, options)?)
            }
            _ => {
                warn!("Unknown provider '{}', using auto-detection", provider_name);
                Self::auto_detect_provider(config)?
            }
        };

        info!("Using {} for transcription", provider.name());

        Ok(Self::from_provider(provider, language))
    }

    pub(crate) fn from_provider(
        provider: Box<dyn TranscriptionProvider>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            language: language.into(),
        }
    }

    fn auto_detect_provider(config: ProviderConfig) -> Result<Box<dyn TranscriptionProvider>> {
        info!("Auto-detecting transcription provider...");

        // Note: OpenAI API requires explicit configuration with api_key
        // Auto-detection skips API providers that need authentication

        if let Ok(provider) =
            OpenAIWhisperCliProvider::new(config.command_path.clone(), "base".to_string())
        {
            if provider.is_available() {
                info!("Auto-detected: OpenAI Whisper CLI");
                return Ok(Box::new(provider));
            }
        }

        let options = config.whisper_cpp_options();
        let model = config.model.clone().unwrap_or_else(|| "base".to_string());
        if let Ok(provider) =
            WhisperCppProvider::new(config.command_path, model, config.model_path, options)
        {
            if provider.is_available() {
                info!("Auto-detected: whisper.cpp");
                return Ok(Box::new(provider));
            }
        }

        Err(anyhow::anyhow!(
            "No transcription provider available. Install whisper-cpp, openai-whisper, or configure OpenAI API with api_key"
        ))
    }

    #[allow(dead_code)] // main.rs compiles this module separately and uses the sample route.
    pub async fn transcribe(&self, audio_path: &PathBuf) -> Result<String> {
        info!(
            "Transcribing audio file: {:?} with {}",
            audio_path,
            self.provider.name()
        );
        bench_trace::event_with_extra("provider_transcription_begin", || {
            serde_json::json!({
                "audio_path": audio_path.display().to_string(),
                "provider": self.provider.name(),
                "language": self.language.as_str(),
            })
        });
        let result = self
            .provider
            .transcribe(audio_path.as_path(), &self.language)
            .await;

        self.trace_transcription_result(&result);
        result
    }

    pub async fn transcribe_samples(
        &self,
        samples: &[f32],
        retention: AudioFileRetention,
    ) -> Result<String> {
        info!(
            "Transcribing {} in-memory samples with {}",
            samples.len(),
            self.provider.name()
        );
        bench_trace::event_with_extra("provider_transcription_begin", || {
            serde_json::json!({
                "provider": self.provider.name(),
                "language": self.language.as_str(),
                "input_kind": "samples",
                "samples": samples.len(),
            })
        });
        let result = self
            .provider
            .transcribe_samples(samples, &self.language, retention)
            .await;

        self.trace_transcription_result(&result);
        result
    }

    fn trace_transcription_result(&self, result: &Result<String>) {
        match &result {
            Ok(text) => bench_trace::event_with_extra("provider_transcription_end", || {
                serde_json::json!({
                    "provider": self.provider.name(),
                    "text_chars": text.len(),
                })
            }),
            Err(error) => bench_trace::event_with_extra("trial_error", || {
                serde_json::json!({
                    "phase": "provider_transcription",
                    "provider": self.provider.name(),
                    "error": error.to_string(),
                })
            }),
        }
    }

    pub async fn prepare(&self) -> Result<()> {
        self.provider.prepare().await
    }

    pub fn supports_chunking(&self) -> bool {
        self.provider.supports_chunking()
    }

    pub fn model_status(&self) -> Option<ModelStatusSnapshot> {
        self.provider.model_status()
    }

    pub fn unload_model(&self) -> Result<()> {
        self.provider.unload_model()
    }

    pub fn recording_complete(&self) -> Result<()> {
        self.provider.recording_complete()
    }

    pub async fn reload_model(&self) -> Result<()> {
        self.provider.reload_model().await
    }

    pub fn is_openai_whisper(&self) -> bool {
        self.provider.name() == "OpenAI Whisper CLI"
    }
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub model: Option<String>,
    pub model_path: Option<String>,
    pub language: Option<String>,
    pub command_path: Option<String>,
    pub api_endpoint: Option<String>,
    pub api_key: Option<String>,
    pub threads: Option<u32>,
    pub beam_size: Option<u32>,
    pub best_of: Option<u32>,
    pub no_fallback: Option<bool>,
    pub timeout_secs: Option<u64>,
    pub keep_warm_for_secs: Option<u64>,
    pub initial_prompt: Option<String>,
    pub coding_vocabulary: Option<String>,
    pub audio_ctx: AudioCtxConfig,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            model: None,
            model_path: None,
            language: Some("en".to_string()),
            command_path: None,
            api_endpoint: None,
            api_key: None,
            threads: None,
            beam_size: None,
            best_of: None,
            no_fallback: None,
            timeout_secs: None,
            keep_warm_for_secs: Some(0),
            initial_prompt: None,
            coding_vocabulary: None,
            audio_ctx: AudioCtxConfig::Auto,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub enum AudioCtxConfig {
    #[default]
    Auto,
    Fixed(u32),
    Full,
}

impl Serialize for AudioCtxConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::Fixed(value) => serializer.serialize_u32(*value),
            Self::Full => serializer.serialize_u32(0),
        }
    }
}

impl<'de> Deserialize<'de> for AudioCtxConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct AudioCtxVisitor;

        impl serde::de::Visitor<'_> for AudioCtxVisitor {
            type Value = AudioCtxConfig;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("auto, off, 0, or a positive audio_ctx integer")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                match value.trim().to_ascii_lowercase().as_str() {
                    "auto" => Ok(AudioCtxConfig::Auto),
                    "off" | "full" => Ok(AudioCtxConfig::Full),
                    other => other
                        .parse::<u32>()
                        .map(audio_ctx_from_u32)
                        .map_err(E::custom),
                }
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let value = u32::try_from(value).map_err(E::custom)?;
                Ok(audio_ctx_from_u32(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let value = u32::try_from(value).map_err(E::custom)?;
                Ok(audio_ctx_from_u32(value))
            }
        }

        deserializer.deserialize_any(AudioCtxVisitor)
    }
}

fn audio_ctx_from_u32(value: u32) -> AudioCtxConfig {
    if value == 0 {
        AudioCtxConfig::Full
    } else {
        AudioCtxConfig::Fixed(value)
    }
}

impl ProviderConfig {
    fn whisper_cpp_options(&self) -> WhisperCppOptions {
        WhisperCppOptions {
            threads: self.threads,
            beam_size: self.beam_size,
            best_of: self.best_of,
            no_fallback: self.no_fallback,
            timeout_secs: self.timeout_secs,
        }
    }

    fn whisper_rs_options(&self) -> WhisperRsOptions {
        WhisperRsOptions {
            threads: self.threads,
            beam_size: self.beam_size,
            best_of: self.best_of,
            no_fallback: self.no_fallback,
            keep_warm_for_secs: self.keep_warm_for_secs,
            initial_prompt: combined_initial_prompt(
                self.initial_prompt.as_deref(),
                self.coding_vocabulary.as_deref(),
            ),
            audio_ctx: self.audio_ctx,
        }
    }
}

fn combined_initial_prompt(
    initial_prompt: Option<&str>,
    coding_vocabulary: Option<&str>,
) -> Option<String> {
    match (initial_prompt, coding_vocabulary) {
        (Some(initial_prompt), Some(coding_vocabulary)) => {
            Some(format!("{initial_prompt}\n{coding_vocabulary}"))
        }
        (Some(initial_prompt), None) => Some(initial_prompt.to_string()),
        (None, Some(coding_vocabulary)) => Some(coding_vocabulary.to_string()),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper::provider::AudioFileRetention;
    use serde::Deserialize;
    use std::future::Future;
    use std::path::Path;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, PartialEq)]
    struct ObservedSamples {
        samples: Vec<f32>,
        language: String,
        retention: AudioFileRetention,
    }

    struct SampleOnlyProvider {
        observed: Arc<Mutex<Option<ObservedSamples>>>,
    }

    impl TranscriptionProvider for SampleOnlyProvider {
        fn name(&self) -> &'static str {
            "sample-only"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn transcribe<'a>(
            &'a self,
            _audio_path: &'a Path,
            _language: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            panic!("sample facade must not call the path transcription route")
        }

        fn transcribe_samples<'a>(
            &'a self,
            samples: &'a [f32],
            language: &'a str,
            retention: AudioFileRetention,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            Box::pin(async move {
                *self.observed.lock().unwrap() = Some(ObservedSamples {
                    samples: samples.to_vec(),
                    language: language.to_string(),
                    retention,
                });
                Ok("in-memory transcript".to_string())
            })
        }
    }

    #[test]
    fn combines_initial_prompt_and_coding_vocabulary() {
        let prompt = combined_initial_prompt(Some("Prefer concise text."), Some("Rust, Tokio"));

        assert_eq!(
            prompt,
            Some("Prefer concise text.\nRust, Tokio".to_string())
        );
    }

    #[test]
    fn provider_config_defaults_to_releasing_the_model_between_recordings() {
        assert_eq!(ProviderConfig::default().keep_warm_for_secs, Some(0));
    }

    #[test]
    fn provider_config_preserves_explicit_positive_keep_warm_duration() {
        let config = ProviderConfig {
            keep_warm_for_secs: Some(300),
            ..Default::default()
        };

        assert_eq!(config.whisper_rs_options().keep_warm_for_secs, Some(300));
    }

    #[derive(Deserialize)]
    struct AudioCtxFixture {
        audio_ctx: AudioCtxConfig,
    }

    #[test]
    fn audio_ctx_config_parses_auto() {
        let fixture: AudioCtxFixture = toml::from_str("audio_ctx = \"auto\"").unwrap();

        assert_eq!(fixture.audio_ctx, AudioCtxConfig::Auto);
    }

    #[test]
    fn audio_ctx_config_parses_fixed_integer() {
        let fixture: AudioCtxFixture = toml::from_str("audio_ctx = 256").unwrap();

        assert_eq!(fixture.audio_ctx, AudioCtxConfig::Fixed(256));
    }

    #[test]
    fn audio_ctx_config_parses_zero_as_full_context() {
        let fixture: AudioCtxFixture = toml::from_str("audio_ctx = 0").unwrap();

        assert_eq!(fixture.audio_ctx, AudioCtxConfig::Full);
    }

    #[test]
    fn audio_ctx_config_parses_off_as_full_context() {
        let fixture: AudioCtxFixture = toml::from_str("audio_ctx = \"off\"").unwrap();

        assert_eq!(fixture.audio_ctx, AudioCtxConfig::Full);
    }

    #[test]
    fn transcriber_reports_pause_chunking_capability() {
        let transcriber = WhisperTranscriber::with_provider(
            "whisper-rs",
            ProviderConfig {
                model_path: Some("/tmp/missing-whisper-model.bin".to_string()),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(transcriber.supports_chunking());
    }

    #[tokio::test]
    async fn sample_facade_forwards_slice_language_and_retention_to_provider_override() {
        let observed = Arc::new(Mutex::new(None));
        let transcriber = WhisperTranscriber {
            provider: Box::new(SampleOnlyProvider {
                observed: observed.clone(),
            }),
            language: "de".to_string(),
        };
        let samples = [0.25, -0.5, 0.75];

        let text = transcriber
            .transcribe_samples(&samples, AudioFileRetention::Keep)
            .await
            .unwrap();

        assert_eq!(text, "in-memory transcript");
        assert_eq!(
            *observed.lock().unwrap(),
            Some(ObservedSamples {
                samples: samples.to_vec(),
                language: "de".to_string(),
                retention: AudioFileRetention::Keep,
            })
        );
    }
}
