use anyhow::Result;
use arboard::Clipboard;
use std::process::{Child, Command, ExitStatus};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

use crate::bench_trace;

const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(10);

fn wait_for_child_with_timeout(child: &mut Child, timeout: Duration) -> Result<ExitStatus> {
    let started = Instant::now();

    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }

        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(anyhow::anyhow!(
                "command timed out after {}ms",
                timeout.as_millis()
            ));
        }

        thread::sleep(COMMAND_POLL_INTERVAL);
    }
}

pub struct ClipboardManager {
    clipboard: Clipboard,
    preserve_previous: bool,
}

impl ClipboardManager {
    pub fn new() -> Result<Self> {
        let clipboard = Clipboard::new()?;

        Ok(Self {
            clipboard,
            preserve_previous: false,
        })
    }

    pub fn with_preserve(mut self, preserve: bool) -> Self {
        self.preserve_previous = preserve;
        self
    }

    pub fn copy_text(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        let previous = if self.preserve_previous {
            self.clipboard.get_text().ok()
        } else {
            None
        };

        info!("Copying {} chars to clipboard", text.len());
        debug!("Text to copy: {}", text);

        self.clipboard.set_text(text)?;

        if let Some(prev) = previous {
            debug!("Previous clipboard content preserved: {} chars", prev.len());
        }

        Ok(())
    }

    pub async fn copy_with_wayland_fallback(&mut self, text: &str) -> Result<()> {
        bench_trace::event_with_extra("clipboard_copy_begin", || {
            serde_json::json!({
                "text_chars": text.len(),
                "preserve_previous": self.preserve_previous,
            })
        });

        // Try arboard first
        let result: Result<()> = (|| {
            if let Err(e) = self.copy_text(text) {
                error!("Arboard clipboard failed: {}, trying wl-copy", e);

                // Fallback to wl-copy command
                use std::io::Write;

                let mut child = Command::new("wl-copy")
                    .stdin(std::process::Stdio::piped())
                    .spawn()?;

                if let Some(mut stdin) = child.stdin.take() {
                    if let Err(e) = stdin.write_all(text.as_bytes()) {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(e.into());
                    }
                }

                let status = wait_for_child_with_timeout(&mut child, COMMAND_TIMEOUT)?;
                if !status.success() {
                    return Err(anyhow::anyhow!("wl-copy failed with status {status}"));
                }

                info!("Copied text using wl-copy fallback");
            }

            Ok(())
        })();

        match &result {
            Ok(()) => bench_trace::event_with_extra("clipboard_copy_end", || {
                serde_json::json!({
                    "success": true,
                    "text_chars": text.len(),
                })
            }),
            Err(error) => bench_trace::event_with_extra("clipboard_copy_end", || {
                serde_json::json!({
                    "success": false,
                    "text_chars": text.len(),
                    "error": error.to_string(),
                })
            }),
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::{Duration, Instant};

    #[test]
    fn wait_for_child_with_timeout_kills_slow_command() {
        let mut child = Command::new("sh")
            .args(["-c", "sleep 2"])
            .spawn()
            .expect("spawn test command");

        let started = Instant::now();
        let err = wait_for_child_with_timeout(&mut child, Duration::from_millis(100))
            .expect_err("slow command should time out");

        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(err.to_string().contains("timed out"));
    }
}
