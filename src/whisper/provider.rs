use anyhow::Result;
use serde::Serialize;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use crate::audio::write_samples_to_wav;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioFileRetention {
    Delete,
    Keep,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelStatusSnapshot {
    pub provider: &'static str,
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub trait TranscriptionProvider: Send + Sync {
    fn name(&self) -> &'static str;

    fn is_available(&self) -> bool;

    fn supports_chunking(&self) -> bool {
        false
    }

    fn prepare<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn model_status(&self) -> Option<ModelStatusSnapshot> {
        None
    }

    fn unload_model(&self) -> Result<()> {
        Err(anyhow::anyhow!(
            "{} does not support explicit model unload",
            self.name()
        ))
    }

    fn recording_complete(&self) -> Result<()> {
        Ok(())
    }

    fn reload_model<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.unload_model()?;
            self.prepare().await
        })
    }

    /// Transcribe mono, 16 kHz, 32-bit float samples.
    ///
    /// File-based providers inherit this adapter, which materializes a uniquely
    /// named WAV and delegates to [`Self::transcribe`].
    fn transcribe_samples<'a>(
        &'a self,
        samples: &'a [f32],
        language: &'a str,
        retention: AudioFileRetention,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let temp_path = tempfile::Builder::new()
                .prefix("chezwizper-")
                .suffix(".wav")
                .tempfile()?
                .into_temp_path();

            match retention {
                AudioFileRetention::Delete => {
                    write_samples_to_wav(temp_path.as_ref(), samples)?;
                    self.transcribe(temp_path.as_ref(), language).await
                }
                AudioFileRetention::Keep => {
                    let path = temp_path.keep()?;
                    write_samples_to_wav(&path, samples)?;
                    tracing::info!("Audio recording retained at: {:?}", path);
                    self.transcribe(&path, language).await
                }
            }
        })
    }

    fn transcribe<'a>(
        &'a self,
        audio_path: &'a Path,
        language: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct ObservedWav {
        path: PathBuf,
        language: String,
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        sample_format: hound::SampleFormat,
        samples: Vec<f32>,
    }

    struct InspectingProvider {
        observations: Mutex<Vec<ObservedWav>>,
        fail: bool,
    }

    impl InspectingProvider {
        fn succeeding() -> Self {
            Self {
                observations: Mutex::new(Vec::new()),
                fail: false,
            }
        }

        fn failing() -> Self {
            Self {
                observations: Mutex::new(Vec::new()),
                fail: true,
            }
        }

        fn observed_paths(&self) -> Vec<PathBuf> {
            self.observations
                .lock()
                .unwrap()
                .iter()
                .map(|observation| observation.path.clone())
                .collect()
        }
    }

    impl TranscriptionProvider for InspectingProvider {
        fn name(&self) -> &'static str {
            "test-provider"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn transcribe<'a>(
            &'a self,
            audio_path: &'a Path,
            language: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            Box::pin(async move {
                let mut reader = hound::WavReader::open(audio_path)?;
                let spec = reader.spec();
                let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
                self.observations.lock().unwrap().push(ObservedWav {
                    path: audio_path.to_path_buf(),
                    language: language.to_string(),
                    channels: spec.channels,
                    sample_rate: spec.sample_rate,
                    bits_per_sample: spec.bits_per_sample,
                    sample_format: spec.sample_format,
                    samples,
                });
                tokio::task::yield_now().await;

                if self.fail {
                    Err(anyhow::anyhow!("injected transcription failure"))
                } else {
                    Ok("delegated transcript".to_string())
                }
            })
        }
    }

    #[tokio::test]
    async fn sample_adapter_delegates_with_float_mono_16khz_wav() {
        let provider = InspectingProvider::succeeding();
        let samples = [0.25, -0.5, 0.75];

        let transcript = provider
            .transcribe_samples(&samples, "fr", AudioFileRetention::Delete)
            .await
            .unwrap();

        assert_eq!(transcript, "delegated transcript");
        let observations = provider.observations.lock().unwrap();
        assert_eq!(observations.len(), 1);
        let observation = &observations[0];
        assert_eq!(observation.language, "fr");
        assert_eq!(observation.channels, 1);
        assert_eq!(observation.sample_rate, 16_000);
        assert_eq!(observation.bits_per_sample, 32);
        assert_eq!(observation.sample_format, hound::SampleFormat::Float);
        assert_eq!(observation.samples, samples);
    }

    #[tokio::test]
    async fn sample_adapter_uses_distinct_paths_for_concurrent_calls() {
        let provider = InspectingProvider::succeeding();

        let (first, second) = tokio::join!(
            provider.transcribe_samples(&[0.1], "en", AudioFileRetention::Delete),
            provider.transcribe_samples(&[0.2], "en", AudioFileRetention::Delete),
        );

        first.unwrap();
        second.unwrap();
        let paths = provider.observed_paths();
        assert_eq!(paths.len(), 2);
        assert_ne!(paths[0], paths[1]);
    }

    #[tokio::test]
    async fn delete_retention_removes_wav_after_success() {
        let provider = InspectingProvider::succeeding();

        provider
            .transcribe_samples(&[0.1], "en", AudioFileRetention::Delete)
            .await
            .unwrap();

        let path = provider.observed_paths().pop().unwrap();
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn delete_retention_removes_wav_after_provider_error() {
        let provider = InspectingProvider::failing();

        let result = provider
            .transcribe_samples(&[0.1], "en", AudioFileRetention::Delete)
            .await;

        assert!(result.is_err());
        let path = provider.observed_paths().pop().unwrap();
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn keep_retention_preserves_valid_wav_after_success() {
        let provider = InspectingProvider::succeeding();

        provider
            .transcribe_samples(&[0.1], "en", AudioFileRetention::Keep)
            .await
            .unwrap();

        let path = provider.observed_paths().pop().unwrap();
        assert!(hound::WavReader::open(&path).is_ok());
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn keep_retention_preserves_valid_wav_after_provider_error() {
        let provider = InspectingProvider::failing();

        let result = provider
            .transcribe_samples(&[0.1], "en", AudioFileRetention::Keep)
            .await;

        assert!(result.is_err());
        let path = provider.observed_paths().pop().unwrap();
        assert!(hound::WavReader::open(&path).is_ok());
        std::fs::remove_file(path).unwrap();
    }
}
