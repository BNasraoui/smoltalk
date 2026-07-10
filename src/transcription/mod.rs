use anyhow::Result;
use std::path::PathBuf;
use tracing::{debug, info};

use crate::bench_trace;
use crate::normalizer::Normalizer;
use crate::whisper::provider::ModelStatusSnapshot;
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

    /// Transcribe audio file and return normalized text
    pub async fn transcribe(&self, audio_path: &PathBuf) -> Result<String> {
        info!("Starting transcription pipeline for: {:?}", audio_path);
        bench_trace::event_with_extra("transcription_begin", || {
            serde_json::json!({
                "audio_path": audio_path.display().to_string(),
            })
        });

        // Step 1: Get raw transcription from whisper
        debug!("Getting raw transcription from whisper");
        let raw_transcription = self.whisper.transcribe(audio_path).await?;
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

        Ok(normalized)
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
    // use super::*;

    #[tokio::test]
    async fn test_transcription_service_creation() {
        //TODO: implement this
        // NOTE:: This would require mocking WhisperTranscriber
    }
}
