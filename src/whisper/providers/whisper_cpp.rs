use anyhow::{Context, Result};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Output, Stdio};
use std::time::Duration;
use tracing::info;
use which::which;

use crate::bench_trace;
use crate::cancellation::CancellationToken;
use crate::whisper::process::{run_command, ProcessError};
use crate::whisper::provider::TranscriptionProvider;

pub struct WhisperCppProvider {
    command_path: PathBuf,
    model_path: Option<String>,
    model: String,
    options: WhisperCppOptions,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WhisperCppOptions {
    pub threads: Option<u32>,
    pub beam_size: Option<u32>,
    pub best_of: Option<u32>,
    pub no_fallback: Option<bool>,
    pub timeout_secs: Option<u64>,
}

impl WhisperCppProvider {
    pub fn new(
        command_path: Option<String>,
        model: String,
        model_path: Option<String>,
        options: WhisperCppOptions,
    ) -> Result<Self> {
        let command_path = if let Some(path) = command_path {
            let custom_path = PathBuf::from(path);
            if custom_path.exists() {
                info!("Using custom whisper.cpp path: {:?}", custom_path);
                custom_path
            } else {
                return Err(anyhow::anyhow!(
                    "Custom whisper path does not exist: {:?}",
                    custom_path
                ));
            }
        } else {
            // Try to find whisper-cli first (as built by our install script), then whisper
            which("whisper-cli")
                .or_else(|_| which("whisper"))
                .context("Whisper CLI not found. Please install whisper.cpp (whisper-cli or whisper command)")?
        };

        info!("Found whisper.cpp at: {:?}", command_path);

        Ok(Self {
            command_path,
            model_path,
            model,
            options,
        })
    }
}

fn add_async_performance_args(cmd: &mut tokio::process::Command, options: &WhisperCppOptions) {
    if let Some(threads) = options.threads {
        cmd.arg("-t").arg(threads.to_string());
    }
    if let Some(beam_size) = options.beam_size {
        cmd.arg("-bs").arg(beam_size.to_string());
    }
    if let Some(best_of) = options.best_of {
        cmd.arg("-bo").arg(best_of.to_string());
    }
    if options.no_fallback.unwrap_or(false) {
        cmd.arg("-nf");
    }
}

async fn run_command_cancellable(
    cmd: tokio::process::Command,
    timeout_secs: Option<u64>,
    context: &'static str,
    cancellation: CancellationToken,
) -> Result<Output> {
    bench_trace::event_with_extra("provider_command_spawn", || {
        serde_json::json!({
            "provider": "whisper.cpp",
            "command": format!("{cmd:?}"),
            "context": context,
        })
    });
    let output = run_command(
        cmd,
        timeout_secs.map(Duration::from_secs),
        context,
        cancellation,
    )
    .await
    .map_err(|error| match error {
        ProcessError::TimedOut => {
            anyhow::anyhow!(
                "whisper.cpp command timed out after {}s",
                timeout_secs.unwrap_or_default()
            )
        }
        error => anyhow::Error::new(error),
    })?;
    bench_trace::event_with_extra("provider_command_exit", || {
        serde_json::json!({
            "provider": "whisper.cpp",
            "status": output.status.code(),
            "success": output.status.success(),
            "context": context,
        })
    });
    Ok(output)
}

impl TranscriptionProvider for WhisperCppProvider {
    fn name(&self) -> &'static str {
        "whisper.cpp"
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
        let model_path = self.model_path.clone();
        let options = self.options.clone();

        Box::pin(async move {
            let model_arg = model_path
                .clone()
                .unwrap_or_else(|| format!("models/ggml-{model}.bin"));
            let mut cmd = tokio::process::Command::new(&command_path);
            cmd.arg("-f")
                .arg(&audio_path)
                .arg("-m")
                .arg(&model_arg)
                .arg("-l")
                .arg(&language)
                .arg("-nt")
                .arg("-np")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .stdin(Stdio::null());
            add_async_performance_args(&mut cmd, &options);

            let output = run_command_cancellable(
                cmd,
                options.timeout_secs,
                "Failed to execute whisper.cpp command",
                cancellation.clone(),
            )
            .await?;

            if !output.status.success() {
                let mut cmd = tokio::process::Command::new(&command_path);
                cmd.arg("-f").arg(&audio_path);
                if let Some(model_path) = &model_path {
                    cmd.arg("-m").arg(model_path);
                }
                add_async_performance_args(&mut cmd, &options);
                cmd.stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .stdin(Stdio::null());

                let output = run_command_cancellable(
                    cmd,
                    options.timeout_secs,
                    "Failed to execute fallback whisper.cpp command",
                    cancellation,
                )
                .await?;
                if !output.status.success() {
                    return Err(anyhow::anyhow!("Whisper.cpp transcription failed"));
                }
                return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }

            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationToken;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    fn unique_test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("chezwizper-{name}-{nanos}"))
    }

    fn write_executable_script(path: &Path, body: &str) {
        fs::write(path, body).unwrap();
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }

    fn read_args(path: &Path) -> Vec<String> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[tokio::test]
    async fn transcribe_passes_optional_performance_flags_to_primary_command() {
        let command_path = unique_test_path("whisper-primary");
        let args_path = unique_test_path("whisper-primary-args");
        let audio_path = unique_test_path("audio.wav");
        fs::write(&audio_path, "audio").unwrap();

        write_executable_script(
            &command_path,
            &format!(
                "#!/bin/sh\nprintf '%s\n' \"$@\" > '{}'\nprintf 'transcript\n'\n",
                args_path.display()
            ),
        );

        let provider = WhisperCppProvider::new(
            Some(command_path.to_string_lossy().to_string()),
            "base".to_string(),
            None,
            WhisperCppOptions {
                threads: Some(8),
                beam_size: Some(5),
                best_of: Some(3),
                no_fallback: Some(true),
                timeout_secs: Some(30),
            },
        )
        .unwrap();

        let result = provider
            .transcribe(&audio_path, "en", CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(result, "transcript");
        assert_eq!(
            read_args(&args_path),
            vec![
                "-f",
                audio_path.to_str().unwrap(),
                "-m",
                "models/ggml-base.bin",
                "-l",
                "en",
                "-nt",
                "-np",
                "-t",
                "8",
                "-bs",
                "5",
                "-bo",
                "3",
                "-nf",
            ]
        );
    }

    #[tokio::test]
    async fn transcribe_passes_optional_performance_flags_to_fallback_command() {
        let command_path = unique_test_path("whisper-fallback");
        let args_path = unique_test_path("whisper-fallback-args");
        let count_path = unique_test_path("whisper-fallback-count");
        let audio_path = unique_test_path("audio.wav");
        fs::write(&audio_path, "audio").unwrap();

        write_executable_script(
            &command_path,
            &format!(
                "#!/bin/sh\ncount=0\nif [ -f '{}' ]; then count=$(cat '{}'); fi\ncount=$((count + 1))\nprintf '%s' \"$count\" > '{}'\nif [ \"$count\" -eq 1 ]; then exit 1; fi\nprintf '%s\n' \"$@\" > '{}'\nprintf 'fallback transcript\n'\n",
                count_path.display(),
                count_path.display(),
                count_path.display(),
                args_path.display()
            ),
        );

        let provider = WhisperCppProvider::new(
            Some(command_path.to_string_lossy().to_string()),
            "base".to_string(),
            Some("custom-model.bin".to_string()),
            WhisperCppOptions {
                threads: Some(4),
                beam_size: Some(2),
                best_of: Some(6),
                no_fallback: Some(true),
                timeout_secs: Some(30),
            },
        )
        .unwrap();

        let result = provider
            .transcribe(&audio_path, "en", CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(result, "fallback transcript");
        assert_eq!(
            read_args(&args_path),
            vec![
                "-f",
                audio_path.to_str().unwrap(),
                "-m",
                "custom-model.bin",
                "-t",
                "4",
                "-bs",
                "2",
                "-bo",
                "6",
                "-nf",
            ]
        );
    }

    #[tokio::test]
    async fn transcribe_times_out_hung_command() {
        let command_path = unique_test_path("whisper-timeout");
        let audio_path = unique_test_path("audio.wav");
        fs::write(&audio_path, "audio").unwrap();

        write_executable_script(&command_path, "#!/bin/sh\nwhile :; do :; done\n");

        let provider = WhisperCppProvider::new(
            Some(command_path.to_string_lossy().to_string()),
            "base".to_string(),
            None,
            WhisperCppOptions {
                timeout_secs: Some(0),
                ..Default::default()
            },
        )
        .unwrap();

        let err = provider
            .transcribe(&audio_path, "en", CancellationToken::new())
            .await
            .unwrap_err();

        assert!(err.to_string().contains("timed out"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_terminates_whisper_cpp_promptly() {
        let command_path = unique_test_path("whisper-cancel");
        let audio_path = unique_test_path("audio.wav");
        fs::write(&audio_path, "audio").unwrap();
        write_executable_script(&command_path, "#!/bin/sh\nexec sleep 1\n");
        let provider = WhisperCppProvider::new(
            Some(command_path.to_string_lossy().to_string()),
            "base".to_string(),
            None,
            WhisperCppOptions::default(),
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        let cancel = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            cancel.cancel();
        });
        let started = Instant::now();

        let result = provider.transcribe(&audio_path, "en", cancellation).await;

        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_millis(500));
    }
}
