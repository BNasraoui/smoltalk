# smoltalk

A small voice dictation daemon for Wayland/Hyprland. Press a keybind (or hold a push-to-talk button), speak, and the transcription is typed into the focused application.

📚 **[View Documentation](./docs/index.md)** — detailed guides and configuration

## Origins

smoltalk is a fork of [ChezWizper](https://github.com/silvabyte/ChezWizper) by silvabyte. Upstream has deprecated ChezWizper in favor of [Audetic](https://github.com/silvabyte/Audetic), its actively maintained successor with a web UI, installer, and macOS support — if you want a fuller product, that's the place to go.

smoltalk instead keeps the small single-daemon design and focuses on transcription latency, with changes verified by a bundled benchmark harness.

> **Naming note:** the binary, systemd service, and config paths are still named `chezwizper` (e.g. `~/.config/chezwizper/config.toml`) while the rename is in progress. Docs refer to the project as smoltalk and to the artifacts by their current literal names.

## Differences from upstream

The upstream design spawns `whisper-cli` and reloads the model from disk for every utterance. smoltalk adds an in-process `whisper-rs` provider. With the default `keep_warm_for_secs = 0`, it starts audio capture, loads `base.en-q5_0` and its inference state during recording, transcribes after release, then unloads the model and state. Measured PSS on the test machine was approximately 250 MiB while loaded and 54 MiB after unloading. Positive `keep_warm_for_secs` values retain the model between recordings. With retention enabled, the bundled harness measured the following results (30 phrases × 3 trials, same machine):

| | whisper-cli subprocess | warm whisper-rs |
|---|---|---|
| Stop-to-text p50 | 3,163 ms | 1,108 ms |
| Stop-to-text p95 | 5,300 ms | 1,280 ms |
| Model load | every utterance | once |

Other changes since the fork:

- Explicit `POST /start` / `POST /stop` endpoints for push-to-talk (bind press/release on a key or mouse button), alongside the original `/toggle`
- Hybrid text injection: single-line text is typed directly without touching the clipboard; multiline text uses a paste transaction that restores the previous clipboard afterwards
- Latency trace instrumentation (`CHEZWIZPER_BENCH_TRACE`) and a benchmark harness under `bench/`
- Configurable API port (`[api] port`), so a benchmark instance can run alongside the live service

## Features

- Keybind toggle and push-to-talk recording
- Local transcription via whisper-rs (in-process and cold while idle by default); whisper.cpp CLI, OpenAI CLI, and OpenAI API providers also available
- Default model lifecycle: approximately 250 MiB PSS while loaded and 54 MiB after unloading on the test machine
- Text injection with clipboard preservation
- Visual recording indicators and Waybar integration
- Single TOML config

## Quick Install (Omarchy + Arch Linux)

```bash
git clone https://github.com/BNasraoui/smoltalk.git
cd smoltalk
make install
```

This installs dependencies, builds with optimized Whisper, sets up services, and configures keybinds.

**After installation:**
1. Start the service: `make start`
2. Toggle mode — add to your Hyprland config:
   `bindd = SUPER, R, smoltalk, exec, curl -X POST http://127.0.0.1:3737/toggle`
3. Or push-to-talk — bind press to `/start` and release to `/stop`:
   ```
   bindd  = , mouse:276, smoltalk start, exec, curl -fsS -X POST http://127.0.0.1:3737/start
   bindrd = , mouse:276, smoltalk stop,  exec, curl -fsS -X POST http://127.0.0.1:3737/stop
   ```

For the in-process provider, set in `~/.config/chezwizper/config.toml`:

```toml
[whisper]
provider = "whisper-rs"
model = "base.en"
```

This uses the cold-at-idle default. To retain the model between recordings, add a positive duration such as `keep_warm_for_secs = 300`.

## Manual Installation

For other distributions or custom setups, see the [Installation Guide](./docs/installation.md).

## Configuration

Default config at `~/.config/chezwizper/config.toml`. See the [Configuration Guide](./docs/configuration.md) for the full reference, including the `[injection]` and `[api]` sections.

## Benchmarking

The daemon emits per-phase JSONL trace events when `CHEZWIZPER_BENCH_TRACE` is set, and `bench/` contains scripts that play a phrase corpus through a PipeWire loopback into a scratch daemon and report per-phase latency, RTF, and memory. See [Benchmarking](./docs/benchmarking.md).

## Development

```bash
make build      # Build debug binary
make release    # Build optimized release
make test       # Run tests
make lint       # Run clippy linter
make fmt        # Check formatting
make fix        # Fix formatting and simple issues

make start      # Enable and start service
make logs       # Show service logs
make restart    # Restart service
make status     # Check service status
make clean      # Clean build artifacts
```

## Troubleshooting

- **Recording issues**: Check the [Configuration Guide](./docs/configuration.md)
- **Text injection fails**: See [Text Injection Setup](./docs/text-injection-setup.md)
- **Service problems**: View logs with `make logs`

## Credits

smoltalk began as [ChezWizper](https://github.com/silvabyte/ChezWizper) by [silvabyte](https://github.com/silvabyte). Their current project, [Audetic](https://github.com/silvabyte/Audetic), continues that codebase as a full-featured product and is worth a look.

## License

MIT
