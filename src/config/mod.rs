use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

pub use crate::vad::VadSettings;
use crate::whisper::AudioCtxConfig;

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub whisper: WhisperConfig,
    pub vad: VadSettings,
    pub chunking: ChunkingConfig,
    pub ui: UiConfig,
    pub wayland: WaylandConfig,
    pub behavior: BehaviorConfig,
    pub injection: InjectionConfig,
    pub api: ApiConfig,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub device: String,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WhisperConfig {
    pub model: String,
    pub language: String,
    pub command_path: Option<String>,
    pub model_path: Option<String>,
    pub api_endpoint: Option<String>,
    pub provider: Option<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkingConfig {
    pub enabled: bool,
    pub pause_ms: u32,
    pub min_chunk_ms: u32,
    pub overlap_ms: u32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pause_ms: 600,
            min_chunk_ms: 5_000,
            overlap_ms: 300,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub indicator_position: String,
    pub indicator_size: u32,
    pub show_notifications: bool,
    pub layer_shell_anchor: String,
    pub layer_shell_margin: u32,
    pub notification_color: String,
    pub waybar: WaybarConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WaybarConfig {
    pub idle_text: String,
    pub recording_text: String,
    pub processing_text: String,
    pub idle_tooltip: String,
    pub recording_tooltip: String,
    pub processing_tooltip: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaylandConfig {
    pub input_method: String,
    pub use_hyprland_ipc: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BehaviorConfig {
    pub auto_paste: bool,
    pub preserve_clipboard: bool,
    pub delete_audio_files: bool,
    #[serde(default = "default_audio_feedback")]
    pub audio_feedback: bool,
}

fn default_audio_feedback() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InjectionConfig {
    pub paste_threshold_chars: usize,
    pub force_method: Option<InjectionForceMethod>,
    pub restore_clipboard: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InjectionForceMethod {
    Type,
    Paste,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    pub port: u16,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            port: 3737, // WHSP in numbers
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: "default".to_string(),
            sample_rate: 16000,
            channels: 1,
        }
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model: "base".to_string(),
            language: "en".to_string(),
            command_path: None,
            model_path: None,
            api_endpoint: Some("https://api.openai.com/v1/audio/transcriptions".to_string()),
            provider: None,
            api_key: None,
            threads: None,
            beam_size: None,
            best_of: None,
            no_fallback: None,
            timeout_secs: None,
            keep_warm_for_secs: Some(300),
            initial_prompt: None,
            coding_vocabulary: None,
            audio_ctx: AudioCtxConfig::Auto,
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            indicator_position: "top-right".to_string(),
            indicator_size: 20,
            show_notifications: true,
            layer_shell_anchor: "top | right".to_string(),
            layer_shell_margin: 10,
            notification_color: "rgb(ff1744)".to_string(),
            waybar: WaybarConfig::default(),
        }
    }
}

impl Default for WaybarConfig {
    fn default() -> Self {
        Self {
            idle_text: "󰑊".to_string(),       // Nerd Font circle with dot (idle)
            recording_text: "󰻃".to_string(),  // Nerd Font record button (recording)
            processing_text: "󰦖".to_string(), // Nerd Font loading/processing icon
            idle_tooltip: "Press Super+R to record".to_string(),
            recording_tooltip: "Recording... Press Super+R to stop".to_string(),
            processing_tooltip: "Processing transcription...".to_string(),
        }
    }
}

impl Default for WaylandConfig {
    fn default() -> Self {
        Self {
            input_method: "wtype".to_string(),
            use_hyprland_ipc: true,
        }
    }
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            auto_paste: true,
            preserve_clipboard: false,
            delete_audio_files: true,
            audio_feedback: true,
        }
    }
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            paste_threshold_chars: 120,
            force_method: None,
            restore_clipboard: true,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;
        Self::load_from_path(config_path)
    }

    pub fn load_from_path(config_path: PathBuf) -> Result<Self> {
        if !config_path.exists() {
            info!(
                "Config file not found, creating default at {:?}",
                config_path
            );
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }

        let content =
            std::fs::read_to_string(&config_path).context("Failed to read config file")?;

        let config: Self = toml::from_str(&content).context("Failed to parse config file")?;

        info!("Loaded config from {:?}", config_path);
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        let content = toml::to_string_pretty(self).context("Failed to serialize config")?;

        std::fs::write(&config_path, content).context("Failed to write config file")?;

        Ok(())
    }

    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().context("Failed to determine config directory")?;

        Ok(config_dir.join("chezwizper").join("config.toml"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::VadEngine;

    #[test]
    fn api_port_defaults_when_section_is_missing() {
        let config: Config = toml::from_str("[audio]\ndevice = \"default\"\n").unwrap();

        assert_eq!(config.api.port, 3737);
    }

    #[test]
    fn api_port_honors_explicit_value() {
        let config: Config = toml::from_str("[api]\nport = 4848\n").unwrap();

        assert_eq!(config.api.port, 4848);
    }

    #[test]
    fn injection_defaults_when_section_is_missing() {
        let config: Config = toml::from_str("[audio]\ndevice = \"default\"\n").unwrap();

        assert_eq!(config.injection.paste_threshold_chars, 120);
        assert_eq!(config.injection.force_method, None);
        assert!(config.injection.restore_clipboard);
    }

    #[test]
    fn injection_honors_explicit_values() {
        let config: Config = toml::from_str(
            "[injection]\npaste_threshold_chars = 42\nforce_method = \"paste\"\nrestore_clipboard = false\n",
        )
        .unwrap();

        assert_eq!(config.injection.paste_threshold_chars, 42);
        assert_eq!(
            config.injection.force_method,
            Some(InjectionForceMethod::Paste)
        );
        assert!(!config.injection.restore_clipboard);
    }

    #[test]
    fn vad_defaults_when_section_is_missing() {
        let config: Config = toml::from_str("[audio]\ndevice = \"default\"\n").unwrap();

        assert!(config.vad.enabled);
        assert_eq!(config.vad.engine, VadEngine::Auto);
        assert_eq!(config.vad.threshold, 0.02);
        assert_eq!(config.vad.min_speech_ms, 100);
        assert_eq!(config.vad.pad_ms, 200);
        assert_eq!(config.vad.model_path, None);
    }

    #[test]
    fn vad_honors_explicit_values() {
        let config: Config = toml::from_str(
            "[vad]\nenabled = false\nengine = \"amplitude\"\nthreshold = 0.04\nmin_speech_ms = 250\npad_ms = 150\nmodel_path = \"/tmp/vad.bin\"\n",
        )
        .unwrap();

        assert!(!config.vad.enabled);
        assert_eq!(config.vad.engine, VadEngine::Amplitude);
        assert_eq!(config.vad.threshold, 0.04);
        assert_eq!(config.vad.min_speech_ms, 250);
        assert_eq!(config.vad.pad_ms, 150);
        assert_eq!(config.vad.model_path, Some(PathBuf::from("/tmp/vad.bin")));
    }

    #[test]
    fn chunking_defaults_are_serialized_when_section_is_missing() {
        let config: Config = toml::from_str("[audio]\ndevice = \"default\"\n").unwrap();
        let value = toml::Value::try_from(config).unwrap();

        assert_eq!(value["chunking"]["enabled"].as_bool(), Some(false));
        assert_eq!(value["chunking"]["pause_ms"].as_integer(), Some(600));
        assert_eq!(value["chunking"]["min_chunk_ms"].as_integer(), Some(5_000));
        assert_eq!(value["chunking"]["overlap_ms"].as_integer(), Some(300));
    }

    #[test]
    fn chunking_honors_explicit_values() {
        let config: Config = toml::from_str(
            "[chunking]\nenabled = true\npause_ms = 800\nmin_chunk_ms = 7000\noverlap_ms = 250\n",
        )
        .unwrap();
        let value = toml::Value::try_from(config).unwrap();

        assert_eq!(value["chunking"]["enabled"].as_bool(), Some(true));
        assert_eq!(value["chunking"]["pause_ms"].as_integer(), Some(800));
        assert_eq!(value["chunking"]["min_chunk_ms"].as_integer(), Some(7_000));
        assert_eq!(value["chunking"]["overlap_ms"].as_integer(), Some(250));
    }
}
