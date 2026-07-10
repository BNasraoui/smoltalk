use anyhow::Result;
use serde::Serialize;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;

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

    fn transcribe<'a>(
        &'a self,
        audio_path: &'a Path,
        language: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}
