use anyhow::Context;
use std::process::{ExitStatus, Output, Stdio};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::cancellation::CancellationToken;

#[derive(Debug, thiserror::Error)]
pub(super) enum ProcessError {
    #[error("transcription cancelled")]
    Cancelled,
    #[error("command timed out")]
    TimedOut,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub(super) async fn run_command(
    mut command: tokio::process::Command,
    timeout: Option<Duration>,
    context: &'static str,
    cancellation: CancellationToken,
) -> std::result::Result<Output, ProcessError> {
    command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .kill_on_drop(true);
    let mut child = command
        .spawn()
        .context(context)
        .map_err(ProcessError::Other)?;
    let stdout = tokio::spawn(read_pipe(child.stdout.take()));
    let stderr = tokio::spawn(read_pipe(child.stderr.take()));

    let timeout = async move {
        match timeout {
            Some(timeout) => tokio::time::sleep(timeout).await,
            None => std::future::pending().await,
        }
    };

    enum Completion {
        Exited(std::io::Result<ExitStatus>),
        Cancelled,
        TimedOut,
    }

    let completion = tokio::select! {
        biased;
        _ = cancellation.cancelled() => Completion::Cancelled,
        _ = timeout => Completion::TimedOut,
        status = child.wait() => Completion::Exited(status),
    };

    let status = match completion {
        Completion::Exited(status) => status.context(context).map_err(ProcessError::Other)?,
        Completion::Cancelled => {
            terminate(&mut child)
                .await
                .context(context)
                .map_err(ProcessError::Other)?;
            let _ = collect_output(stdout, stderr).await;
            return Err(ProcessError::Cancelled);
        }
        Completion::TimedOut => {
            terminate(&mut child)
                .await
                .context(context)
                .map_err(ProcessError::Other)?;
            let _ = collect_output(stdout, stderr).await;
            return Err(ProcessError::TimedOut);
        }
    };
    let (stdout, stderr) = collect_output(stdout, stderr)
        .await
        .map_err(ProcessError::Other)?;

    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

async fn terminate(child: &mut tokio::process::Child) -> std::io::Result<()> {
    match child.kill().await {
        Ok(()) => Ok(()),
        Err(error) => match child.try_wait()? {
            Some(_) => Ok(()),
            None => Err(error),
        },
    }
}

async fn read_pipe<R>(pipe: Option<R>) -> std::io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut output = Vec::new();
    if let Some(mut pipe) = pipe {
        pipe.read_to_end(&mut output).await?;
    }
    Ok(output)
}

async fn collect_output(
    stdout: tokio::task::JoinHandle<std::io::Result<Vec<u8>>>,
    stderr: tokio::task::JoinHandle<std::io::Result<Vec<u8>>>,
) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
    let stdout = stdout.await.context("stdout reader task failed")??;
    let stderr = stderr.await.context("stderr reader task failed")??;
    Ok((stdout, stderr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationToken;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    #[tokio::test]
    async fn cancellation_waits_until_the_child_is_reaped() {
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("slow-command");
        let pid_path = directory.path().join("pid");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nprintf '%s' $$ > '{}'\nexec sleep 5\n",
                pid_path.display()
            ),
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script, permissions).unwrap();
        let command = tokio::process::Command::new(&script);
        let cancellation = CancellationToken::new();
        let cancel = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel.cancel();
        });

        let result = run_command(command, None, "test command failed", cancellation).await;

        assert!(result.is_err());
        let pid = std::fs::read_to_string(pid_path).unwrap();
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }
}
