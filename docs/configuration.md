# smoltalk Configuration Guide

smoltalk is configured via a single TOML file located at `~/.config/chezwizper/config.toml`. This guide covers everything you need to know about configuring smoltalk for your needs.

## Quick Start

The minimal configuration to get started:

```toml
[whisper]
# Recommended: the warm in-process provider (fastest by a wide margin)
provider = "whisper-rs"
model = "base.en"
language = "en"
```

smoltalk will create a default configuration file on first run if none exists.

## Complete Configuration Example

Here's a full configuration file with all available options:

```toml
[audio]
device = "default"              # Audio input device name
sample_rate = 16000             # Sample rate in Hz (8000, 16000, 44100, 48000)
channels = 1                    # Number of audio channels (1 = mono, 2 = stereo)

[whisper]
provider = "whisper-rs"         # Transcription provider (see Providers section)
model = "base.en"               # Model name (provider-specific)
language = "en"                 # Language code (ISO 639-1)
keep_warm_for_secs = 300        # whisper-rs: unload the model after this much idle time
audio_ctx = "auto"              # whisper-rs: shrink encoder context to clip length (big speedup for short clips)
# initial_prompt = "Transcribe concise technical dictation."   # whisper-rs: bias decoding
# coding_vocabulary = "Rust, Axum, Tokio"                      # whisper-rs: appended to the prompt
# threads = 1                   # Local providers: decoding threads
# beam_size = 1                 # Local providers: lower = faster
# best_of = 1                   # Local providers: lower = faster
# no_fallback = true            # whisper.cpp: faster, less robust on noisy audio
# timeout_secs = 120            # whisper.cpp: kill a wedged transcription
# command_path = "/usr/bin/whisper"  # CLI providers: custom tool path
# model_path = "/path/to/model.bin"  # Custom model file path
# api_key = "sk-your-api-key-here"   # openai-api only
# api_endpoint = "https://api.openai.com/v1/audio/transcriptions"  # openai-api only

[api]
port = 3737                     # Local HTTP API port (toggle/start/stop/status)

[injection]
paste_threshold_chars = 120     # Short single-line text below this is typed directly
restore_clipboard = true        # Restore previous clipboard after a paste injection
# force_method = "type"         # Optional: always "type" or always "paste"

[ui]
indicator_position = "top-right"  # Visual indicator position
indicator_size = 20             # Indicator size in pixels
show_notifications = true       # Show desktop notifications
layer_shell_anchor = "top | right"  # Wayland layer shell anchor
layer_shell_margin = 10         # Margin from screen edge in pixels

[ui.waybar]
idle_text = "󰑊"                # Icon shown when idle (ready to record)
recording_text = "󰻃"           # Icon shown when recording
processing_text = "󰦖"          # Icon shown when processing transcription
idle_tooltip = "Press Super+R to record"                    # Tooltip for idle state
recording_tooltip = "Recording... Press Super+R to stop"     # Tooltip for recording state
processing_tooltip = "Processing transcription..."           # Tooltip for processing state

[wayland]
input_method = "wtype"          # Text injection method
use_hyprland_ipc = true         # Use Hyprland IPC for better integration

[behavior]
auto_paste = true               # Automatically paste transcribed text
preserve_clipboard = false      # Keep clipboard content after pasting
delete_audio_files = true       # Delete temporary audio files after processing
audio_feedback = true           # Play audio feedback sounds
```

## Configuration Sections

### [audio] - Audio Input Settings

Controls how smoltalk captures audio from your microphone.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `device` | string | `"default"` | Audio input device name. Use `"default"` for system default, or specific device name |
| `sample_rate` | number | `16000` | Audio sample rate in Hz. Common values: 8000, 16000, 44100, 48000 |
| `channels` | number | `1` | Number of audio channels. 1 = mono (recommended), 2 = stereo |

**Tips:**
- 16000 Hz sample rate provides the best balance of quality and performance for speech
- Mono (1 channel) is sufficient for speech recognition and reduces file size
- To list available audio devices: `arecord -l` (on Linux)

### [whisper] - Transcription Settings

Configures speech-to-text transcription providers and models.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | string | auto-detect | Transcription provider: `"whisper-rs"`, `"whisper-cpp"`, `"openai-cli"`, `"openai-api"`, or omit for auto-detection |
| `model` | string | `"base"` | Model name (provider-specific, see Providers section) |
| `language` | string | `"en"` | Language code (ISO 639-1 format) |
| `keep_warm_for_secs` | number | none | whisper-rs: unload the model after this much idle time (stays warm forever if unset) |
| `audio_ctx` | string/number | `"auto"` | whisper-rs: `"auto"` shrinks the encoder context to the clip length; `0`/`"off"` uses the full 30s window; an integer sets it explicitly |
| `initial_prompt` | string | none | whisper-rs: prompt to bias decoding (casing, punctuation, domain terms) |
| `coding_vocabulary` | string | none | whisper-rs: comma-separated terms appended to the prompt |
| `threads` | number | provider default | Local providers: decoding threads |
| `beam_size` | number | provider default | Local providers: beam width (lower = faster) |
| `best_of` | number | provider default | Local providers: candidate count (lower = faster) |
| `no_fallback` | bool | `false` | whisper.cpp: disable temperature fallback (faster, less robust) |
| `timeout_secs` | number | none | whisper.cpp: kill the subprocess after this long |
| `command_path` | string | auto-detect | CLI providers: custom path to the whisper tool |
| `model_path` | string | auto-detect | Custom path to the model file (whisper-rs / whisper.cpp) |
| `api_key` | string | none | openai-api: API key (required) |
| `api_endpoint` | string | OpenAI API | openai-api: custom endpoint URL |

#### Providers

smoltalk supports multiple transcription providers:

**whisper-rs** (`provider = "whisper-rs"`) — **recommended**
- **Best for:** Lowest latency — this is smoltalk's reason to exist
- **How it works:** Loads the model in-process once and keeps it warm; no subprocess spawn or model reload per utterance (measured stop-to-text p50 ~1.1s vs ~3.2s for the CLI path)
- **Requirements:** A ggml model file (the installer's models work; prefers quantized variants like `ggml-base.en-q5_0.bin` when present)
- **Models:** `"tiny.en"`, `"base.en"`, `"small.en"`, and multilingual equivalents
- **Memory:** Model stays resident (~150 MB for base.en); use `keep_warm_for_secs` to unload when idle
- **Note:** Not yet part of auto-detection — set it explicitly

**OpenAI API** (`provider = "openai-api"`)
- **Best for:** High accuracy, no local setup
- **Requirements:** API key in config, internet connection  
- **Models:** `"whisper-1"` (only available model)
- **Cost:** ~$0.006 per minute of audio

**OpenAI Whisper CLI** (`provider = "openai-cli"`)
- **Best for:** Local processing, no API costs, privacy
- **Requirements:** `pip install openai-whisper`
- **Models:** `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"`
- **Cost:** Free (local processing)

**whisper.cpp** (`provider = "whisper-cpp"`)
- **Best for:** Resource-constrained systems, CPU-only inference
- **Requirements:** Build from source or install via package manager
- **Models:** `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large"`
- **Status:** Experimental
- **Cost:** Free (local processing)

**Auto-Detection** (omit `provider`)
- smoltalk automatically selects from the CLI providers:
  1. OpenAI Whisper CLI (if installed)
  2. whisper.cpp (fallback)
- Note: `whisper-rs` (the fastest option) and API providers must be configured explicitly

#### Language Codes

Common language codes (ISO 639-1):

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `en` | English | `es` | Spanish | `fr` | French |
| `de` | German | `it` | Italian | `pt` | Portuguese |
| `ru` | Russian | `zh` | Chinese | `ja` | Japanese |
| `ko` | Korean | `ar` | Arabic | `auto` | Auto-detect* |

*Auto-detection only works with OpenAI API

For the complete list, see [ISO 639-1 codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

### [ui] - User Interface Settings

Controls visual indicators and desktop notifications.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `indicator_position` | string | `"top-right"` | Screen position: `"top-left"`, `"top-right"`, `"bottom-left"`, `"bottom-right"` |
| `indicator_size` | number | `20` | Visual indicator size in pixels |
| `show_notifications` | bool | `true` | Show desktop notifications for transcription results |
| `layer_shell_anchor` | string | `"top \| right"` | Wayland layer shell anchor points |
| `layer_shell_margin` | number | `10` | Distance from screen edge in pixels |

#### [ui.waybar] - Waybar Integration

Customize icons and tooltips for Waybar status display. See [Waybar Integration](./waybar-integration.md) for setup instructions.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `idle_text` | string | `"󰑊"` | Icon shown when idle (ready to record) - Nerd Font icon |
| `recording_text` | string | `"󰻃"` | Icon shown when actively recording - Nerd Font icon |
| `processing_text` | string | `"󰦖"` | Icon shown when processing transcription - Nerd Font icon |
| `idle_tooltip` | string | `"Press Super+R to record"` | Tooltip text when hovering over idle state |
| `recording_tooltip` | string | `"Recording... Press Super+R to stop"` | Tooltip text when hovering during recording |
| `processing_tooltip` | string | `"Processing transcription..."` | Tooltip text when processing audio |

**Icon Tips:**
- Uses Nerd Font icons for consistency with other Waybar modules
- Icons inherit colors from your Waybar theme via CSS classes
- All styling is controlled by CSS, not inline styles
- Custom icons can be any Unicode character or Nerd Font glyph

### [wayland] - Wayland Integration

Configures integration with Wayland desktop environments.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `input_method` | string | `"wtype"` | Text injection method: `"wtype"`, `"clipboard"` |
| `use_hyprland_ipc` | bool | `true` | Use Hyprland IPC for better window management integration |

**Text Injection Methods:**
- `"wtype"` - Direct text typing (fast, works in most apps)
- `"clipboard"` - Via clipboard (universal compatibility, slower)

### [injection] - Hybrid Text Injection

Controls how transcribed text lands in the focused application. Short single-line text is typed directly so your clipboard is never touched; longer or multiline text uses a guarded paste transaction (save clipboard → set transcript → paste → restore). See [Text Injection Setup](./text-injection-setup.md) for details.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `paste_threshold_chars` | number | `120` | Single-line text at or below this length is typed directly; anything longer (or containing a newline) is pasted |
| `force_method` | string | none | Force `"type"` or `"paste"` for all text, bypassing the hybrid decision |
| `restore_clipboard` | bool | `true` | Restore the previous clipboard contents after a paste injection. Best-effort for plain text; rich/image/clipboard-manager state cannot be perfectly preserved by wl-clipboard/X11 tools |

### [api] - HTTP API

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `port` | number | `3737` | Local HTTP port for `/toggle`, `/start`, `/stop`, and `/status`. Change it to run a second instance (the benchmark harness uses 3838) |

### [behavior] - Application Behavior

Controls how smoltalk handles transcribed text and temporary files.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_paste` | bool | `true` | Automatically paste/type transcribed text |
| `preserve_clipboard` | bool | `false` | Keep existing clipboard content when using clipboard injection |
| `delete_audio_files` | bool | `true` | Delete temporary audio recordings after processing |
| `audio_feedback` | bool | `true` | Play audio feedback sounds (start/stop recording) |

## Configuration File Location

smoltalk looks for its configuration file at:

- **Linux:** `~/.config/chezwizper/config.toml`
- **macOS:** `~/Library/Application Support/chezwizper/config.toml`
- **Windows:** `%APPDATA%\chezwizper\config.toml`

## Environment Variables

smoltalk respects these environment variables:

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Logging level (`error`, `warn`, `info`, `debug`, `trace`) |
| `CHEZWIZPER_BENCH_TRACE` | Path to a JSONL file; when set, the daemon emits per-phase latency trace events (see [Benchmarking](./benchmarking.md)) |

## Common Configuration Scenarios

### For OpenAI API Users
```toml
[whisper]
provider = "openai-api"
api_key = "sk-your-api-key-here"  # Your OpenAI API key
model = "whisper-1"
language = "en"  # or "auto" for automatic detection
```

### For Local Processing (Privacy-Focused, Fastest)
```toml
[whisper]
provider = "whisper-rs"
model = "base.en"    # Loads once, stays warm; ~1.1s stop-to-text
language = "en"
audio_ctx = "auto"   # Shrink encoder work for short clips

# No API key needed - everything runs locally
```

### For Multiple Languages
```toml
[whisper]
provider = "openai-api"
model = "whisper-1" 
language = "auto"  # Automatically detect language

# Or set a specific language code like "es" for Spanish
```

### For Low-Resource Systems
```toml
[audio]
sample_rate = 16000  # Lower sample rate
channels = 1         # Mono audio

[whisper]
provider = "openai-cli"
model = "tiny"       # Smallest, fastest model
language = "en"

[behavior]
delete_audio_files = true  # Clean up temp files
```

### For High Accuracy Transcription
```toml
[audio]
sample_rate = 48000  # Higher quality audio
channels = 1

[whisper]
provider = "openai-cli"
model = "large-v3"   # Most accurate model
language = "en"

[behavior]
audio_feedback = false  # Reduce distractions
```

## Migrating from Earlier Versions

If you're upgrading from an earlier version that used `use_api = true/false`, update your config:

**Old format:**
```toml
[whisper]
use_api = true
model = "whisper-1"
```

**New format:**
```toml
[whisper]
provider = "openai-api"
model = "whisper-1"
```

**Migration mapping:**
- `use_api = true` → `provider = "openai-api"`
- `use_api = false` → `provider = "openai-cli"` (or `"whisper-cpp"`)

## Troubleshooting Configuration

### Config File Issues

**"Failed to parse config file"**
- Check TOML syntax with an online validator
- Ensure strings are quoted: `language = "en"` not `language = en`
- Verify boolean values: `true`/`false` not `"true"`/`"false"`

**"Config file not found"**
- smoltalk will create a default config on first run
- Manually create the config directory: `mkdir -p ~/.config/chezwizper`

### Provider Issues

**"No transcription provider available"**
- Install a provider: `pip install openai-whisper` 
- Or set OpenAI API key: `export OPENAI_API_KEY="sk-..."`
- Check provider installation: `whisper --help`

**"OPENAI_API_KEY environment variable required"**
- Set your API key: `export OPENAI_API_KEY="sk-your-key"`
- Get an API key from https://platform.openai.com/api-keys

### Audio Issues

**"No audio input detected"**
- Check `device = "default"` in config
- List devices: `arecord -l`
- Test audio: `arecord -f cd test.wav` (Ctrl+C to stop, `aplay test.wav` to playback)

### Validation

Test your configuration:
```bash
# Start smoltalk with verbose logging
RUST_LOG=debug chezwizper

# Look for these log messages:
# "Loaded config from ..."
# "Using [Provider] for transcription"
```

For more troubleshooting, see the [Installation Guide](./installation.md) and [Text Injection Setup](./text-injection-setup.md).