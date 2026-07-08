use anyhow::Result;
use regex::Regex;
use std::sync::OnceLock;
use tracing::{debug, info};

/// Trait for normalizing transcription output from various whisper implementations
pub trait TranscriptionNormalizer: Send + Sync {
    /// Normalize the raw transcription output
    fn normalize(&self, raw_output: &str) -> String;

    /// Get the name of this normalizer for logging
    fn name(&self) -> &'static str;
}

/// Normalizer for whisper.cpp output format
pub struct WhisperCppNormalizer {
    timestamp_regex: Regex,
}

impl WhisperCppNormalizer {
    pub fn new() -> Result<Self> {
        // Matches timestamps like [00:00:00.000 --> 00:00:03.280] or [00:00:00:000 --> 00:00:03:280]
        let timestamp_regex =
            Regex::new(r"\[\d{2}:\d{2}:\d{2}[:.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[:.]\d{3}\]\s*")?;

        Ok(Self { timestamp_regex })
    }
}

impl TranscriptionNormalizer for WhisperCppNormalizer {
    fn normalize(&self, raw_output: &str) -> String {
        debug!("Normalizing whisper.cpp output");

        let mut cleaned = String::new();

        for line in raw_output.lines() {
            // Remove timestamps from the beginning of lines
            let line_cleaned = self.timestamp_regex.replace_all(line, "");
            let line_trimmed = line_cleaned.trim();

            // Skip empty lines
            if !line_trimmed.is_empty() {
                if !cleaned.is_empty() {
                    cleaned.push(' ');
                }
                cleaned.push_str(line_trimmed);
            }
        }

        let result = strip_non_speech_markers(&cleaned);
        debug!(
            "Normalized {} chars to {} chars",
            raw_output.len(),
            result.len()
        );

        result
    }

    fn name(&self) -> &'static str {
        "WhisperCppNormalizer"
    }
}

/// Normalizer for OpenAI Whisper output format
pub struct OpenAIWhisperNormalizer;

impl Default for OpenAIWhisperNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAIWhisperNormalizer {
    pub fn new() -> Self {
        Self
    }
}

impl TranscriptionNormalizer for OpenAIWhisperNormalizer {
    fn normalize(&self, raw_output: &str) -> String {
        // OpenAI Whisper typically outputs clean text without timestamps
        // Just trim whitespace and remove non-speech markers.
        strip_non_speech_markers(raw_output)
    }

    fn name(&self) -> &'static str {
        "OpenAIWhisperNormalizer"
    }
}

fn strip_non_speech_markers(text: &str) -> String {
    static MARKER_REGEX: OnceLock<Regex> = OnceLock::new();
    static WHITESPACE_REGEX: OnceLock<Regex> = OnceLock::new();

    let marker_regex = MARKER_REGEX.get_or_init(|| {
        Regex::new(
            r"(?ix)
            (?:
                \[
                    \s*
                    (?:blank_audio|silence|silent|no \s+ speech|inaudible|music|noise|background \s+ noise)
                    \s*
                \]
                |
                \(
                    \s*
                    (?:blank_audio|silence|silent|no \s+ speech|inaudible|music|noise|background \s+ noise)
                    \s*
                \)
            )",
        )
        .expect("non-speech marker regex is valid")
    });
    let whitespace_regex =
        WHITESPACE_REGEX.get_or_init(|| Regex::new(r"\s+").expect("whitespace regex is valid"));

    let stripped = marker_regex.replace_all(text, " ");
    whitespace_regex
        .replace_all(stripped.trim(), " ")
        .trim()
        .to_string()
}

/// Enum to hold different normalizer types
pub enum Normalizer {
    WhisperCpp(WhisperCppNormalizer),
    OpenAIWhisper(OpenAIWhisperNormalizer),
}

impl Normalizer {
    /// Create a normalizer based on whether this is OpenAI whisper or whisper.cpp
    pub fn create(is_openai_whisper: bool) -> Result<Self> {
        if is_openai_whisper {
            info!("Creating OpenAI Whisper normalizer");
            Ok(Normalizer::OpenAIWhisper(OpenAIWhisperNormalizer::new()))
        } else {
            info!("Creating whisper.cpp normalizer");
            Ok(Normalizer::WhisperCpp(WhisperCppNormalizer::new()?))
        }
    }

    /// Run normalization using the appropriate normalizer
    pub fn run(&self, raw_output: &str) -> String {
        match self {
            Normalizer::WhisperCpp(n) => {
                debug!("Running {}", n.name());
                n.normalize(raw_output)
            }
            Normalizer::OpenAIWhisper(n) => {
                debug!("Running {}", n.name());
                n.normalize(raw_output)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_cpp_normalizer() {
        let normalizer = WhisperCppNormalizer::new().unwrap();

        let input = "[00:00:00.000 --> 00:00:03.280] This is me talking\n[00:00:03.280 --> 00:00:05.000] And more text";
        let expected = "This is me talking And more text";

        assert_eq!(normalizer.normalize(input), expected);
    }

    #[test]
    fn test_whisper_cpp_normalizer_with_colons() {
        let normalizer = WhisperCppNormalizer::new().unwrap();

        let input = "[00:00:00:000 --> 00:00:03:280] This is me talking";
        let expected = "This is me talking";

        assert_eq!(normalizer.normalize(input), expected);
    }

    #[test]
    fn test_openai_whisper_normalizer() {
        let normalizer = OpenAIWhisperNormalizer::new();

        let input = "  This is clean text  ";
        let expected = "This is clean text";

        assert_eq!(normalizer.normalize(input), expected);
    }

    #[test]
    fn non_speech_markers_only_normalize_to_empty_text() {
        let normalizer = OpenAIWhisperNormalizer::new();

        assert_eq!(normalizer.normalize(" [BLANK_AUDIO] "), "");
        assert_eq!(normalizer.normalize(" [silence]\n(silence) "), "");
    }

    #[test]
    fn non_speech_markers_are_stripped_from_real_text() {
        let normalizer = WhisperCppNormalizer::new().unwrap();

        let input = "[00:00:00.000 --> 00:00:03.280] hello [BLANK_AUDIO] world (silence)";

        assert_eq!(normalizer.normalize(input), "hello world");
    }
}
