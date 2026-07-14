use anyhow::{Context, Result};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Command;
use tracing::info;
use which::which;

use crate::bench_trace;
use crate::cancellation::CancellationToken;
use crate::whisper::process::run_command;
use crate::whisper::provider::TranscriptionProvider;

pub struct OpenAIWhisperCliProvider {
    command_path: PathBuf,
    model: String,
}

impl OpenAIWhisperCliProvider {
    pub fn new(command_path: Option<String>, model: String) -> Result<Self> {
        let command_path = if let Some(path) = command_path {
            let custom_path = PathBuf::from(path);
            if custom_path.exists() {
                info!("Using custom OpenAI whisper path: {:?}", custom_path);
                custom_path
            } else {
                return Err(anyhow::anyhow!(
                    "Custom whisper path does not exist: {:?}",
                    custom_path
                ));
            }
        } else {
            which("whisper")
                .context("OpenAI Whisper CLI not found. Please install openai-whisper")?
        };

        let help_output = Command::new(&command_path).arg("--help").output();

        let is_openai = if let Ok(output) = help_output {
            let help_text = String::from_utf8_lossy(&output.stdout);
            help_text.contains("--output_format") && help_text.contains("--output_dir")
        } else {
            false
        };

        if !is_openai {
            return Err(anyhow::anyhow!(
                "Detected whisper CLI is not OpenAI Whisper"
            ));
        }

        info!("Detected OpenAI Whisper CLI at: {:?}", command_path);

        Ok(Self {
            command_path,
            model,
        })
    }
}

impl TranscriptionProvider for OpenAIWhisperCliProvider {
    fn name(&self) -> &'static str {
        "OpenAI Whisper CLI"
    }

    fn is_available(&self) -> bool {
        self.command_path.exists()
    }

    fn transcribe<'a>(
        &'a self,
        audio_path: &'a Path,
        language: &'a str,
        cancellation: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        let audio_path = audio_path.to_path_buf();
        let language = language.to_string();
        let command_path = self.command_path.clone();
        let model = self.model.clone();

        Box::pin(async move {
            let audio_stem = audio_path
                .file_stem()
                .context("Invalid audio path")?
                .to_str()
                .context("Invalid audio filename")?;
            let output_path = PathBuf::from(format!("/tmp/{audio_stem}.txt"));
            let mut command = tokio::process::Command::new(&command_path);
            command
                .arg(&audio_path)
                .arg("--model")
                .arg(&model)
                .arg("--language")
                .arg(&language)
                .arg("--output_format")
                .arg("txt")
                .arg("--output_dir")
                .arg("/tmp");

            bench_trace::event_with_extra("provider_command_spawn", || {
                serde_json::json!({
                    "provider": "OpenAI Whisper CLI",
                    "command": format!("{command:?}"),
                    "model": model,
                    "language": language,
                })
            });

            let output = match run_command(
                command,
                None,
                "Failed to execute whisper command",
                cancellation,
            )
            .await
            {
                Ok(output) => output,
                Err(error) => {
                    let _ = tokio::fs::remove_file(&output_path).await;
                    return Err(error.into());
                }
            };
            bench_trace::event_with_extra("provider_command_exit", || {
                serde_json::json!({
                    "provider": "OpenAI Whisper CLI",
                    "status": output.status.code(),
                    "success": output.status.success(),
                })
            });

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let _ = tokio::fs::remove_file(&output_path).await;
                return Err(anyhow::anyhow!("Whisper transcription failed: {}", stderr));
            }

            let transcription = tokio::fs::read_to_string(&output_path)
                .await
                .context("Failed to read transcription output");
            let _ = tokio::fs::remove_file(&output_path).await;

            Ok(transcription?.trim().to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationToken;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_terminates_the_cli_process_promptly() {
        let directory = tempfile::tempdir().unwrap();
        let command_path = directory.path().join("slow-whisper");
        std::fs::write(&command_path, "#!/bin/sh\nexec sleep 1\n").unwrap();
        let mut permissions = std::fs::metadata(&command_path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&command_path, permissions).unwrap();
        let provider = OpenAIWhisperCliProvider {
            command_path,
            model: "base.en".to_string(),
        };
        let cancellation = CancellationToken::new();
        let cancel = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            cancel.cancel();
        });
        let started = Instant::now();

        let result = provider
            .transcribe(Path::new("unused.wav"), "en", cancellation)
            .await;

        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_millis(500));
    }
}
