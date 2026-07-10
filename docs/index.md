# smoltalk Documentation

Welcome to the smoltalk documentation. smoltalk is a lean, latency-first voice dictation daemon for Wayland/Hyprland, hard-forked from [ChezWizper](https://github.com/silvabyte/ChezWizper) (now deprecated upstream — its maintained successor is [Audetic](https://github.com/silvabyte/Audetic)).

> **Naming note:** the binary, systemd service, and config paths still use the `chezwizper` name while the rename is in progress. Commands and paths in these docs are the current literal names.

## Available Documentation

### Installation & Setup

- [Installation Guide](./installation.md) — complete installation instructions
- [Configuration Guide](./configuration.md) — full configuration reference: providers (including the in-process `whisper-rs` provider), audio, injection, API, UI, and behavior
- [Text Injection Setup](./text-injection-setup.md) — how hybrid injection works and how to set up injection tools
- [Waybar Integration](./waybar-integration.md) — status indicators in your Waybar

### Performance

- [Benchmarking](./benchmarking.md) — the trace sink, the benchmark harness, and how to measure changes before shipping them
- [Pause-Triggered Chunking Experiment](./chunking-experiment.md) — design, initial latency results, thermal tradeoffs, and adoption gate

### Development

- [Architecture](./architecture.md) — module map, data flow from keypress to injected text, and the design behind the latency-sensitive paths
- [Adding Providers](./adding-providers.md) — guide for adding new transcription providers

smoltalk uses a Makefile for common development tasks:

```bash
make help       # Show all available commands
make build      # Build debug binary
make release    # Build optimized release
make test       # Run tests
make lint       # Run clippy linter
make fmt        # Check formatting
make start      # Enable and start service
make logs       # Show service logs
make status     # Check service status
```

## HTTP API

The daemon listens on `127.0.0.1:3737` by default (configurable via `[api] port`):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/toggle` | POST | Toggle recording (start if idle, stop-and-transcribe if recording) |
| `/start` | POST | Start recording (idempotent; no-op unless idle) — for push-to-talk press |
| `/stop` | POST | Stop and transcribe (idempotent; no-op unless recording) — for push-to-talk release |
| `/status` | GET | JSON status: `{"recording": bool, "status": "idle\|recording\|processing"}` |

## Quick Links

- [Main README](../README.md) — project overview and quick start
- [GitHub Repository](https://github.com/BNasraoui/smoltalk) — source code and issues
- [ChezWizper](https://github.com/silvabyte/ChezWizper) (upstream, deprecated) and [Audetic](https://github.com/silvabyte/Audetic) (its maintained successor)
