use anyhow::Result;
use tracing::{debug, info};

use crate::bench_trace;
use crate::cancellation::CancellationToken;
use crate::normalizer::Normalizer;
use crate::whisper::provider::{AudioFileRetention, ModelStatusSnapshot};
use crate::whisper::WhisperTranscriber;

/// Service that orchestrates transcription and normalization
pub struct TranscriptionService {
    whisper: WhisperTranscriber,
    normalizer: Normalizer,
}

impl TranscriptionService {
    /// Create a new transcription service with the provided whisper transcriber
    pub fn new(whisper: WhisperTranscriber) -> Result<Self> {
        let normalizer = Normalizer::create(whisper.is_openai_whisper())?;

        Ok(Self {
            whisper,
            normalizer,
        })
    }

    /// Transcribe mono, 16 kHz, 32-bit float samples and return normalized text.
    pub async fn transcribe_samples(
        &self,
        samples: &[f32],
        retention: AudioFileRetention,
        cancellation: CancellationToken,
    ) -> Result<String> {
        info!(
            "Starting cancellable transcription pipeline for {} in-memory samples",
            samples.len()
        );
        bench_trace::event_with_extra("transcription_begin", || {
            serde_json::json!({
                "input_kind": "samples",
                "samples": samples.len(),
            })
        });

        let raw_transcription = self
            .whisper
            .transcribe_samples(samples, retention, cancellation.clone())
            .await?;
        if cancellation.is_cancelled() {
            return Err(anyhow::anyhow!("transcription cancelled"));
        }

        Ok(self.normalize_transcription(raw_transcription))
    }

    fn normalize_transcription(&self, raw_transcription: String) -> String {
        bench_trace::event_with_extra("transcription_raw_done", || {
            serde_json::json!({
                "raw_chars": raw_transcription.len(),
            })
        });

        // Step 2: Normalize the transcription
        debug!("Normalizing transcription output");
        let normalized = self.normalizer.run(&raw_transcription);
        bench_trace::event_with_extra("normalization_done", || {
            serde_json::json!({
                "raw_chars": raw_transcription.len(),
                "normalized_chars": normalized.len(),
            })
        });
        bench_trace::event_with_extra("transcription_end", || {
            serde_json::json!({
                "text_chars": normalized.len(),
            })
        });

        info!(
            "Transcription pipeline complete: {} chars -> {} chars",
            raw_transcription.len(),
            normalized.len()
        );

        normalized
    }

    pub async fn prepare(&self) -> Result<()> {
        self.whisper.prepare().await
    }

    pub fn supports_chunking(&self) -> bool {
        self.whisper.supports_chunking()
    }

    pub fn model_status(&self) -> Option<ModelStatusSnapshot> {
        self.whisper.model_status()
    }

    pub fn unload_model(&self) -> Result<()> {
        self.whisper.unload_model()
    }

    pub fn recording_complete(&self) -> Result<()> {
        self.whisper.recording_complete()
    }

    pub async fn reload_model(&self) -> Result<()> {
        self.whisper.reload_model().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationToken;
    use crate::whisper::provider::{AudioFileRetention, TranscriptionProvider};
    use std::future::Future;
    use std::path::Path;
    use std::pin::Pin;

    struct RawSampleProvider;

    impl TranscriptionProvider for RawSampleProvider {
        fn name(&self) -> &'static str {
            "raw-sample-provider"
        }

        fn is_available(&self) -> bool {
            true
        }

        fn transcribe<'a>(
            &'a self,
            _audio_path: &'a Path,
            _language: &'a str,
            _cancellation: CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            panic!("sample pipeline must not call the path route")
        }

        fn transcribe_samples<'a>(
            &'a self,
            _samples: &'a [f32],
            _language: &'a str,
            _retention: AudioFileRetention,
            _cancellation: CancellationToken,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            Box::pin(async {
                Ok("[00:00:00.000 --> 00:00:01.000] hello [BLANK_AUDIO] world".to_string())
            })
        }
    }

    #[tokio::test]
    async fn sample_pipeline_normalizes_provider_output() {
        let whisper = WhisperTranscriber::from_provider(Box::new(RawSampleProvider), "en");
        let service = TranscriptionService::new(whisper).unwrap();

        let text = service
            .transcribe_samples(
                &[0.1, 0.2],
                AudioFileRetention::Delete,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(text, "hello world");
    }

    #[tokio::test]
    async fn cancelled_sample_pipeline_does_not_return_a_transcript() {
        let whisper = WhisperTranscriber::from_provider(Box::new(RawSampleProvider), "en");
        let service = TranscriptionService::new(whisper).unwrap();
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        let result = service
            .transcribe_samples(&[0.1, 0.2], AudioFileRetention::Delete, cancellation)
            .await;

        assert!(result.is_err());
    }
}
