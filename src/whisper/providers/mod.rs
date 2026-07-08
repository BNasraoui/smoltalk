pub mod openai_api;
pub mod openai_cli;
pub mod whisper_cpp;
pub mod whisper_rs;

pub use openai_api::OpenAIProvider;
pub use openai_cli::OpenAIWhisperCliProvider;
pub(crate) use whisper_cpp::WhisperCppOptions;
pub use whisper_cpp::WhisperCppProvider;
pub(crate) use whisper_rs::WhisperRsOptions;
pub use whisper_rs::WhisperRsProvider;
