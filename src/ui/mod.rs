use crate::config::UiConfig;
use anyhow::{Context, Result};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

const HYPRCTL_TIMEOUT: Duration = Duration::from_secs(1);
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(10);

fn command_output_with_timeout(command: &mut Command, timeout: Duration) -> Result<Output> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = command.spawn().context("Failed to spawn command")?;
    wait_for_child_with_timeout(&mut child, timeout)?;

    child
        .wait_with_output()
        .context("Failed to collect command output")
}

fn wait_for_child_with_timeout(child: &mut Child, timeout: Duration) -> Result<ExitStatus> {
    let started = Instant::now();

    loop {
        if let Some(status) = child.try_wait().context("Failed to check command status")? {
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

pub struct Indicator {
    audio_feedback_enabled: bool,
    notifications_enabled: bool,
    notification_color: String,
}

impl Default for Indicator {
    fn default() -> Self {
        Self::new()
    }
}

impl Indicator {
    pub fn new() -> Self {
        Self {
            audio_feedback_enabled: true,
            notifications_enabled: true,
            notification_color: "rgb(ff1744)".to_string(),
        }
    }

    pub fn from_config(config: &UiConfig) -> Self {
        Self {
            audio_feedback_enabled: true,
            notifications_enabled: config.show_notifications,
            notification_color: config.notification_color.clone(),
        }
    }

    pub fn with_audio_feedback(mut self, enabled: bool) -> Self {
        self.audio_feedback_enabled = enabled;
        self
    }

    pub async fn show_recording(&self) -> Result<()> {
        info!("Showing recording indicator");

        if let Err(e) = self.hyprland_notify("󰻃 Recording...") {
            debug!("Hyprland notification failed: {}", e);
        }

        // Play recording start sound
        self.play_sound("start").await;

        Ok(())
    }

    pub async fn show_processing(&self) -> Result<()> {
        info!("Showing processing indicator");

        if let Err(e) = self.hyprland_notify("󰦖 Processing...") {
            debug!("Hyprland notification failed: {}", e);
        }

        // Play recording stop sound
        self.play_sound("stop").await;

        Ok(())
    }

    pub async fn show_complete(&self, text: &str) -> Result<()> {
        info!("Showing completion indicator");

        let mut chars = text.chars();
        let preview: String = chars.by_ref().take(50).collect();
        let preview = if chars.next().is_some() {
            format!("{preview}...")
        } else {
            preview
        };

        if let Err(e) = self.hyprland_notify(&format!("󰸞 {preview}")) {
            debug!("Hyprland notification failed: {}", e);
        }

        // Play completion sound
        self.play_sound("complete").await;

        Ok(())
    }

    pub async fn show_cancelled(&self) -> Result<()> {
        info!("Showing cancellation indicator");

        if let Err(e) = self.hyprland_notify("󰜺 Recording cancelled") {
            debug!("Hyprland notification failed: {}", e);
        }

        self.play_sound("cancel").await;
        Ok(())
    }

    pub async fn show_error(&self, error: &str) -> Result<()> {
        warn!("Showing error: {}", error);

        if let Err(e) = self.hyprland_notify(&format!("Error: {error}")) {
            debug!("Hyprland notification failed: {}", e);
        }

        Ok(())
    }

    fn hyprland_notify(&self, title: &str) -> Result<()> {
        if !self.notifications_enabled {
            return Ok(());
        }

        let mut command = Command::new("hyprctl");
        command.args(["notify", "-1", "3000", &self.notification_color, title]);
        command_output_with_timeout(&mut command, HYPRCTL_TIMEOUT)?;

        Ok(())
    }

    async fn play_sound(&self, sound_type: &str) {
        if !self.audio_feedback_enabled {
            return;
        }

        debug!("Playing {} sound", sound_type);

        // Use a simple approach with system commands
        let sound_type = sound_type.to_string();
        tokio::spawn(async move {
            if let Err(e) = Self::play_simple_sound(&sound_type).await {
                debug!("Failed to play sound: {}", e);
            }
        });
    }

    async fn play_simple_sound(sound_type: &str) -> Result<()> {
        let (freq, duration_ms) = match sound_type {
            "start" => (800, 150),     // High pitch, short beep
            "stop" => (400, 200),      // Low pitch, longer beep
            "complete" => (1000, 100), // Very high pitch, very short beep
            "cancel" => (250, 120),    // Low, short cancellation cue
            _ => (500, 150),
        };

        // Try generating custom beep tones first (more distinctive)
        if let Ok(output) = Self::generate_beep_tone(freq, duration_ms).await {
            if output.status.success() || output.status.code() == Some(124) {
                debug!(
                    "Played {} with generated tone ({}Hz, {}ms)",
                    sound_type, freq, duration_ms
                );
                return Ok(());
            }
        }

        // Fallback to system sounds if tone generation fails
        let sound_files = vec![
            "/usr/share/sounds/alsa/Front_Left.wav",
            "/usr/share/sounds/freedesktop/stereo/bell.oga",
            "/usr/share/sounds/Oxygen-Sys-Log-In.ogg",
        ];

        for sound_file in sound_files {
            if std::path::Path::new(sound_file).exists() {
                if let Ok(output) = Command::new("aplay").arg(sound_file).output() {
                    if output.status.success() {
                        debug!("Played {} with aplay: {}", sound_type, sound_file);
                        return Ok(());
                    }
                }
            }
        }

        debug!("No working sound method found for {}", sound_type);
        Ok(())
    }

    async fn generate_beep_tone(freq: u32, duration_ms: u32) -> Result<std::process::Output> {
        // Try different methods to generate custom beep tones

        // Method 1: Use speaker-test (if available)
        let duration_secs = format!("{:.1}", duration_ms as f64 / 1000.0);
        if let Ok(output) = Command::new("timeout")
            .args([
                &duration_secs,
                "speaker-test",
                "-t",
                "sine",
                "-f",
                &freq.to_string(),
                "-c",
                "1",
            ])
            .output()
        {
            if output.status.success() || output.status.code() == Some(124) {
                // 124 = timeout success
                return Ok(output);
            }
        }

        // Method 2: Use beep command (if available)
        if let Ok(output) = Command::new("beep")
            .args(["-f", &freq.to_string(), "-l", &duration_ms.to_string()])
            .output()
        {
            return Ok(output);
        }

        // Method 3: Generate tone with paplay + Python
        let python_cmd = format!(
            "python3 -c \"
import math, sys
samples = int(44100 * {duration_ms} / 1000)
freq = {freq}
for i in range(samples):
    t = i / 44100.0
    sample = math.sin(2.0 * math.pi * freq * t) * 0.3
    sample_i16 = int(sample * 16384)
    sys.stdout.buffer.write(sample_i16.to_bytes(2, 'little', signed=True))
\" | paplay --raw --format=s16le --rate=44100 --channels=1"
        );

        if let Ok(output) = Command::new("bash").args(["-c", &python_cmd]).output() {
            return Ok(output);
        }

        Err(anyhow::anyhow!("No tone generation method available"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_honors_show_notifications() {
        let config = UiConfig {
            show_notifications: false,
            ..Default::default()
        };

        let indicator = Indicator::from_config(&config);

        assert!(!indicator.notifications_enabled);
    }

    #[tokio::test]
    async fn show_complete_handles_long_non_ascii_text() {
        let config = UiConfig {
            show_notifications: false,
            ..Default::default()
        };
        let indicator = Indicator::from_config(&config).with_audio_feedback(false);
        let text = "語".repeat(60);

        indicator.show_complete(&text).await.unwrap();
    }
}
