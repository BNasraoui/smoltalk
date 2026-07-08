#!/usr/bin/env python3
"""Collect end-to-end ChezWizper traces and derive trial rows."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from summarize import main as summarize_main


PHASES = {
    "api_toggle_received",
    "main_toggle_dequeued",
    "audio_first_sample",
    "audio_stop_begin",
    "samples_taken",
    "wav_write_done",
    "transcription_begin",
    "transcription_end",
    "clipboard_copy_begin",
    "clipboard_copy_end",
    "injection_begin",
    "injection_end",
    "state_idle_set",
    "trial_error",
}


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_toml(path):
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def dump_toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(dump_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


def write_toml(path, data):
    lines = []
    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    for key, value in sorted(scalars.items()):
        lines.append(f"{key} = {dump_toml_value(value)}")
    for section, values in sorted((k, v) for k, v in data.items() if isinstance(v, dict)):
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in sorted(values.items()):
            if isinstance(value, dict):
                continue
            lines.append(f"{key} = {dump_toml_value(value)}")
        for child, child_values in sorted((k, v) for k, v in values.items() if isinstance(v, dict)):
            lines.append("")
            lines.append(f"[{section}.{child}]")
            for key, value in sorted(child_values.items()):
                lines.append(f"{key} = {dump_toml_value(value)}")
    path.write_text("\n".join(lines) + "\n")


@dataclass(frozen=True)
class DaemonConfig:
    path: Path
    port: int
    api_url: str

    @classmethod
    def from_base(cls, base_path, output_path, port=3838, measure_injection=False):
        config = load_toml(Path(base_path).expanduser())
        config.setdefault("api", {})["port"] = port
        config.setdefault("behavior", {})["auto_paste"] = bool(measure_injection)
        config["behavior"]["audio_feedback"] = False
        config["behavior"]["delete_audio_files"] = True
        config.setdefault("ui", {})["show_notifications"] = False
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_toml(output, config)
        return cls(output, port, f"http://127.0.0.1:{port}")


class IncrementalJsonlReader:
    def __init__(self, path, offset=0):
        self.path = Path(path)
        self.offset = offset

    def read_new(self):
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                start = handle.tell()
                raw = handle.readline()
                if not raw:
                    self.offset = start
                    return rows
                if not raw.endswith(b"\n"):
                    self.offset = start
                    return rows
                self.offset = handle.tell()
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSONL in {self.path} at byte {start}: {line[:120]}") from exc


def http_json(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_toggle(api_url):
    request = urllib.request.Request(f"{api_url}/toggle", method="POST")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        response.read()


def wait_for_idle(api_url, timeout_seconds=30):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if http_json(f"{api_url}/status").get("status") == "idle":
                return True
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    return False


def ms_between(events, start, end):
    start_index = next((i for i, e in enumerate(events) if e.get("event") == start), None)
    if start_index is None:
        return None
    last = next((e for e in events[start_index + 1:] if e.get("event") == end), None)
    if not last:
        return None
    return round((last["monotonic_ns"] - events[start_index]["monotonic_ns"]) / 1_000_000, 3)


def first_event(events, name):
    return next((event for event in events if event.get("event") == name), None)


def samples_taken(events):
    event = first_event(events, "samples_taken")
    samples = event.get("extra", {}).get("samples") if event else None
    return samples if isinstance(samples, (int, float)) else None


def text_source(events):
    if first_event(events, "injection_end"):
        return "injection"
    if first_event(events, "clipboard_copy_end"):
        return "clipboard"
    return "idle"


@dataclass(frozen=True)
class ResourceSnapshot:
    rss_kb: int
    hwm_kb: int
    user_ticks: int
    system_ticks: int


@dataclass(frozen=True)
class ResourceStats:
    peak_rss_mb: float
    cpu_user_ms: float
    cpu_system_ms: float

    @classmethod
    def empty(cls):
        return cls(peak_rss_mb=0.0, cpu_user_ms=0.0, cpu_system_ms=0.0)

    @classmethod
    def from_snapshots(cls, first, last, clock_ticks=None):
        if first is None or last is None:
            return cls.empty()
        ticks = clock_ticks or ProcTreeSampler.clock_ticks_default()
        return cls(
            peak_rss_mb=round(last.hwm_kb / 1024, 3),
            cpu_user_ms=round((last.user_ticks - first.user_ticks) * 1000 / ticks, 3),
            cpu_system_ms=round((last.system_ticks - first.system_ticks) * 1000 / ticks, 3),
        )


class ProcTreeSampler:
    def __init__(self, pid, proc_root=Path("/proc"), clock_ticks=None):
        self.pid = int(pid)
        self.proc_root = Path(proc_root)
        self.clock_ticks = clock_ticks or self.clock_ticks_default()

    @staticmethod
    def clock_ticks_default():
        return os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def process_tree(self):
        seen = set()
        pending = [self.pid]
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            if not (self.proc_root / str(pid)).exists():
                continue
            seen.add(pid)
            task_dir = self.proc_root / str(pid) / "task"
            for child_file in task_dir.glob("*/children"):
                try:
                    pending.extend(int(value) for value in child_file.read_text().split())
                except OSError:
                    continue
        return seen

    def snapshot(self):
        rss_kb = 0
        hwm_kb = 0
        user_ticks = 0
        system_ticks = 0
        for pid in self.process_tree():
            status = self._read_status(pid)
            rss_kb += status.get("VmRSS", 0)
            hwm_kb += status.get("VmHWM", status.get("VmRSS", 0))
            stat = self._read_stat(pid)
            user_ticks += stat[0]
            system_ticks += stat[1]
        return ResourceSnapshot(rss_kb=rss_kb, hwm_kb=hwm_kb, user_ticks=user_ticks, system_ticks=system_ticks)

    def _read_status(self, pid):
        values = {}
        try:
            for line in (self.proc_root / str(pid) / "status").read_text().splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, rest = line.split(":", 1)
                    values[key] = int(rest.split()[0])
        except (OSError, ValueError):
            pass
        return values

    def _read_stat(self, pid):
        try:
            raw = (self.proc_root / str(pid) / "stat").read_text()
            after_comm = raw.rsplit(")", 1)[1].split()
            return int(after_comm[11]), int(after_comm[12])
        except (OSError, IndexError, ValueError):
            return 0, 0


class BackgroundResourceSampler:
    def __init__(self, pid, interval_seconds=0.05):
        self.sampler = ProcTreeSampler(pid)
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snapshots = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1)

    def _run(self):
        while not self._stop.is_set():
            snapshot = self.sampler.snapshot()
            with self._lock:
                self._snapshots.append(snapshot)
            time.sleep(self.interval_seconds)

    def mark(self):
        with self._lock:
            return len(self._snapshots)

    def stats_since(self, index):
        with self._lock:
            snapshots = self._snapshots[index:]
        if len(snapshots) < 2:
            return ResourceStats.empty()
        first = snapshots[0]
        last = snapshots[-1]
        peak_hwm = max(snapshot.hwm_kb for snapshot in snapshots)
        last = ResourceSnapshot(last.rss_kb, peak_hwm, last.user_ticks, last.system_ticks)
        return ResourceStats.from_snapshots(first, last, self.sampler.clock_ticks)


class AudioSource:
    def daemon_env(self):
        return {}

    def verify(self):
        return None

    def record_once(self, phrase):
        raise NotImplementedError

    def close(self):
        return None


class SleepAudioSource(AudioSource):
    def __init__(self, seconds):
        self.seconds = seconds

    def record_once(self, phrase):
        duration = phrase.get("duration_ms")
        time.sleep((duration / 1000.0) if duration else self.seconds)


class PlaybackAudioSource(AudioSource):
    def __init__(self, sink_name="chezwizper_bench", rate=16000, tail_padding_ms=300):
        self.sink_name = sink_name
        self.rate = rate
        self.tail_padding_ms = tail_padding_ms
        self.module_id = None
        self.previous_default_source = None

    @classmethod
    def available(cls):
        return all(shutil.which(command) for command in ("pactl", "paplay", "parecord"))

    def __enter__(self):
        proc = subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={self.sink_name}", f"rate={self.rate}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "unknown pactl error").strip()
            raise RuntimeError(f"failed to create Pulse/PipeWire null sink: {detail}")
        self.module_id = proc.stdout.strip()
        # Force unity gain end to end: an attenuated sink/monitor feeds the
        # daemon quiet audio, which degrades both VAD and WER measurements.
        subprocess.run(["pactl", "set-sink-volume", self.sink_name, "100%"], check=False)
        subprocess.run(
            ["pactl", "set-source-volume", f"{self.sink_name}.monitor", "100%"], check=False
        )
        subprocess.run(["pactl", "set-sink-mute", self.sink_name, "0"], check=False)
        # The daemon records via cpal's ALSA default, which on PipeWire systems
        # follows the DEFAULT SOURCE and ignores PULSE_SOURCE. Flip the default
        # source to the bench monitor for the run and restore it afterwards.
        # (Consequence: any OTHER app starting a recording during the run also
        # hears the bench audio — don't dictate while a benchmark is running.)
        got = subprocess.run(
            ["pactl", "get-default-source"], capture_output=True, text=True, check=False
        )
        self.previous_default_source = got.stdout.strip() or None
        subprocess.run(
            ["pactl", "set-default-source", f"{self.sink_name}.monitor"], check=True
        )
        print(
            "note: default audio source redirected to the bench sink for this run; "
            "it is restored on exit",
            file=sys.stderr,
        )
        self._install_cleanup_handlers()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _install_cleanup_handlers(self):
        # A SIGTERM/SIGINT mid-run must still restore the user's default
        # source and unload the sink — otherwise every new recording stream
        # on the system (including live dictation) hears bench silence.
        import atexit

        atexit.register(self.close)
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous = signal.getsignal(signum)

            def handler(sig, frame, previous=previous):
                self.close()
                if callable(previous):
                    previous(sig, frame)
                else:
                    raise SystemExit(128 + sig)

            signal.signal(signum, handler)

    def daemon_env(self):
        return {"PULSE_SOURCE": f"{self.sink_name}.monitor"}

    def verify(self):
        # parecord targets the monitor explicitly; pw-record must NOT be used
        # here — it ignores PULSE_SOURCE and silently records the microphone.
        # Capture raw s16le: parecord only finalizes a WAV header on clean
        # exit, and we stop it with SIGTERM.
        with tempfile.NamedTemporaryFile(suffix=".raw") as capture, tempfile.NamedTemporaryFile(suffix=".wav") as tone:
            self._write_tone(Path(tone.name))
            recorder = subprocess.Popen(
                [
                    "parecord",
                    f"--device={self.sink_name}.monitor",
                    f"--rate={self.rate}",
                    "--channels=1",
                    "--format=s16le",
                    "--raw",
                    # Default fragsize is 2s of audio; short captures would be
                    # killed before the first fragment ever reaches the file.
                    "--latency-msec=100",
                    capture.name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            subprocess.run(["paplay", "--device", self.sink_name, tone.name], check=True)
            time.sleep(0.3)
            recorder.terminate()
            try:
                recorder.wait(timeout=2)
            except subprocess.TimeoutExpired:
                recorder.kill()
                recorder.wait()
            data = Path(capture.name).read_bytes()
            if not any(byte != 0 for byte in data):
                raise RuntimeError("loopback verification captured silence")

    def record_once(self, phrase):
        wav_path = phrase.get("wav_path")
        if not wav_path:
            raise RuntimeError(f"phrase {phrase.get('id')} has no wav_path")
        subprocess.run(["paplay", "--device", self.sink_name, wav_path], check=True)
        time.sleep(self.tail_padding_ms / 1000.0)

    def close(self):
        if getattr(self, "previous_default_source", None):
            subprocess.run(
                ["pactl", "set-default-source", self.previous_default_source], check=False
            )
            self.previous_default_source = None
        if self.module_id:
            subprocess.run(["pactl", "unload-module", self.module_id], check=False)
            self.module_id = None

    def _write_tone(self, path):
        import math
        import wave

        # 1s: monitors only emit frames while a stream is playing, and the
        # recorder needs time to connect before the tone ends.
        frames = bytearray()
        for i in range(int(self.rate * 1.0)):
            sample = int(12000 * math.sin(2 * math.pi * 440 * i / self.rate))
            frames.extend(sample.to_bytes(2, "little", signed=True))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.rate)
            wav.writeframes(bytes(frames))


def wav_has_signal(path):
    import wave

    try:
        with wave.open(str(path), "rb") as wav:
            data = wav.readframes(wav.getnframes())
    except (OSError, wave.Error):
        return False
    return any(byte != 0 for byte in data)


def open_audio_source(args):
    if args.audio_source == "sleep":
        return SleepAudioSource(args.record_seconds)
    if args.audio_source == "loopback" and not PlaybackAudioSource.available():
        raise RuntimeError("loopback audio source requires pactl, paplay, and pw-record")
    if args.audio_source == "loopback" or PlaybackAudioSource.available():
        source = PlaybackAudioSource(tail_padding_ms=args.tail_padding_ms)
        source.__enter__()
        try:
            source.verify()
        except Exception:
            source.close()
            raise
        return source
    print("warning: pactl/paplay/pw-record unavailable; falling back to sleep audio source", file=sys.stderr)
    return SleepAudioSource(args.record_seconds)


def bench_event(run_id, event, phrase=None, trial_id=None):
    extra = {}
    if phrase:
        extra["phrase_id"] = phrase["id"]
    if trial_id:
        extra["trial_id"] = trial_id
    return {
        "schema": "chezwizper-bench-event-v1",
        "run_id": run_id,
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "extra": extra,
    }


def build_trial_row(run_id, trial_id, phrase, trial_events, provider, model, model_path, threads, resources):
    failure = first_event(trial_events, "trial_error")
    audio_samples = samples_taken(trial_events)
    audio_ms = round(audio_samples * 1000 / 16000, 3) if audio_samples is not None else phrase.get("duration_ms")
    transcription_ms = ms_between(trial_events, "transcription_begin", "transcription_end")
    source = text_source(trial_events)
    total_end = {
        "injection": "injection_end",
        "clipboard": "clipboard_copy_end",
        "idle": "state_idle_set",
    }[source]

    if failure:
        status = "failed"
    elif not trial_events:
        status = "no_events"
    elif not any(e.get("event") == "audio_stop_begin" for e in trial_events):
        status = "attribution_error"
    else:
        status = "ok"

    return {
        "run_id": run_id,
        "trial_id": trial_id,
        "phrase_id": phrase["id"],
        "phrase_group": phrase["group"],
        "scenario": "e2e",
        "provider": provider,
        "engine_version": None,
        "model": model,
        "model_path": model_path,
        "threads": threads,
        "audio_ms": audio_ms,
        "press_to_recording_start_ms": ms_between(trial_events, "api_toggle_received", "audio_first_sample"),
        "stop_to_wav_ms": ms_between(trial_events, "audio_stop_begin", "wav_write_done"),
        "wav_to_transcription_ms": transcription_ms,
        "transcription_to_injection_ms": ms_between(trial_events, "transcription_end", "injection_end"),
        "total_stop_to_text_ms": ms_between(trial_events, "audio_stop_begin", total_end),
        "total_stop_to_text_source": source,
        "rtf": round((transcription_ms / 1000) / (audio_samples / 16000), 4)
        if transcription_ms is not None and audio_samples
        else None,
        "wer": None,
        "cer": None,
        "peak_rss_mb": resources.peak_rss_mb,
        "cpu_user_ms": resources.cpu_user_ms,
        "cpu_system_ms": resources.cpu_system_ms,
        "cpu_temp_c": read_max_cpu_temp(),
        "text": None,
        "status": status,
        "error": failure.get("extra", {}).get("error") if failure else None,
    }


def read_max_cpu_temp():
    """Max core temperature in °C, or None. Thermal throttling silently
    invalidates latency comparisons — every trial records the temp so hot
    runs are identifiable after the fact."""
    temps = []
    for zone in Path("/sys/class/hwmon").glob("hwmon*/temp*_input"):
        try:
            temps.append(int(zone.read_text().strip()) / 1000)
        except (OSError, ValueError):
            continue
    return round(max(temps), 1) if temps else None


def collect_until_idle(reader, deadline):
    events = []
    while time.time() < deadline:
        events.extend(reader.read_new())
        idle_index = next((i for i, e in enumerate(events) if e.get("event") == "state_idle_set"), None)
        if idle_index is not None:
            return [e for e in events[: idle_index + 1] if e.get("event") in PHASES]
        time.sleep(0.05)
    return [e for e in events if e.get("event") in PHASES]


def run_trials(args, daemon, audio_source, events_path, trials_path, run_id):
    phrases = read_jsonl(Path(args.corpus))
    reader = IncrementalJsonlReader(events_path, events_path.stat().st_size if events_path.exists() else 0)
    with BackgroundResourceSampler(daemon.pid) as resources:
        for phrase in phrases:
            for trial in range(1, args.warm_trials + 1):
                trial_id = f"{phrase['id']}-{trial}"
                if not wait_for_idle(args.api_url):
                    raise RuntimeError(f"daemon not idle before trial {trial_id}")

                write_jsonl(events_path, bench_event(run_id, "bench_trial_begin", phrase, trial_id))
                reader.read_new()
                resource_mark = resources.mark()
                try:
                    post_toggle(args.api_url)
                    audio_source.record_once(phrase)
                    post_toggle(args.api_url)
                    trial_events = collect_until_idle(reader, time.time() + args.trial_timeout)
                except Exception as exc:
                    trial_events = collect_until_idle(reader, time.time() + 5)
                    trial_events.append({
                        "event": "trial_error",
                        "monotonic_ns": time.monotonic_ns(),
                        "extra": {"error": str(exc)},
                    })
                write_jsonl(events_path, bench_event(run_id, "bench_trial_end", phrase, trial_id))

                row = build_trial_row(
                    run_id=run_id,
                    trial_id=trial_id,
                    phrase=phrase,
                    trial_events=trial_events,
                    provider=args.provider,
                    model=args.model,
                    model_path=args.model_path,
                    threads=args.threads,
                    resources=resources.stats_since(resource_mark),
                )
                write_jsonl(trials_path, row)


def summarize(output_dir, trials_path):
    old_argv = sys.argv
    try:
        sys.argv = ["summarize.py", "--trials", str(trials_path), "--output-dir", str(output_dir)]
        summarize_main()
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="bench-artifacts/corpus/corpus.jsonl")
    parser.add_argument("--output-dir", default="bench-artifacts")
    parser.add_argument("--daemon", default="target/release/chezwizper")
    parser.add_argument("--base-config", default=str(Path.home() / ".config/chezwizper/config.toml"))
    parser.add_argument("--generated-config")
    parser.add_argument("--api-port", type=int, default=3838)
    parser.add_argument("--api-url")
    parser.add_argument("--measure-injection", action="store_true")
    parser.add_argument("--warm-trials", type=int, default=3)
    parser.add_argument("--record-seconds", type=float, default=1.2)
    parser.add_argument("--tail-padding-ms", type=int, default=300)
    parser.add_argument("--trial-timeout", type=int, default=180)
    parser.add_argument("--audio-source", choices=("auto", "loopback", "sleep"), default="auto")
    parser.add_argument("--provider", default="chezwizper")
    parser.add_argument("--model", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--threads", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    trials_path = output_dir / "trials.jsonl"
    config_path = Path(args.generated_config) if args.generated_config else output_dir / "bench-config.toml"
    events_path.write_text("")
    trials_path.write_text("")

    config = DaemonConfig.from_base(args.base_config, config_path, args.api_port, args.measure_injection)
    args.api_url = args.api_url or config.api_url
    run_id = time.strftime("%Y%m%dT%H%M%S")

    audio_source = open_audio_source(args)
    env = os.environ.copy()
    env.update(audio_source.daemon_env())
    env["CHEZWIZPER_BENCH_TRACE"] = str(events_path)
    env["CHEZWIZPER_BENCH_RUN_ID"] = run_id

    daemon = subprocess.Popen([args.daemon, "--config", str(config.path)], env=env)
    try:
        if not wait_for_idle(args.api_url, timeout_seconds=30):
            raise RuntimeError(f"daemon did not become idle at {args.api_url}")
        run_trials(args, daemon, audio_source, events_path, trials_path, run_id)
    finally:
        try:
            audio_source.close()
        finally:
            daemon.send_signal(signal.SIGTERM)
            try:
                daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
            summarize(output_dir, trials_path)


if __name__ == "__main__":
    main()
