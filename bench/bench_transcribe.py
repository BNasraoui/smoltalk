#!/usr/bin/env python3
"""Run offline transcription benchmarks over the corpus WAV fixtures."""

import argparse
import json
import os
import resource
import shlex
import subprocess
import time
from pathlib import Path

from summarize import main as summarize_main


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def corpus_uses_silence_fallback(rows):
    return any(row.get("tts") == "silence-fallback" for row in rows)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def error_rates(expected, actual):
    expected_words = expected.lower().split()
    actual_words = actual.lower().split()
    wer = edit_distance(expected_words, actual_words) / max(1, len(expected_words))
    cer = edit_distance(expected.lower(), actual.lower()) / max(1, len(expected))
    return round(wer, 4), round(cer, 4)


def render_command(template, phrase):
    values = {
        "wav": phrase["wav_path"],
        "phrase_id": phrase["id"],
        "expected_text": phrase["expected_text"],
    }
    return template.format(**values)


def run_trial(command_template, phrase, timeout):
    started = time.monotonic_ns()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)

    if command_template:
        command = render_command(command_template, phrase)
        proc = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        text = proc.stdout.strip()
        status = "ok" if proc.returncode == 0 else "failed"
        error = proc.stderr.strip() if proc.returncode else None
    else:
        text = ""
        status = "skipped"
        error = "no --command-template supplied"

    ended = time.monotonic_ns()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    user_ms = round((usage_after.ru_utime - usage_before.ru_utime) * 1000, 3)
    system_ms = round((usage_after.ru_stime - usage_before.ru_stime) * 1000, 3)
    elapsed_ms = round((ended - started) / 1_000_000, 3)
    peak_rss_mb = round(usage_after.ru_maxrss / 1024, 3)

    return text, status, error, elapsed_ms, user_ms, system_ms, peak_rss_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="bench-artifacts/corpus/corpus.jsonl")
    parser.add_argument("--output-dir", default="bench-artifacts")
    parser.add_argument("--command-template", help="Example: whisper-cli -m model.bin -f {wav} -nt -np")
    parser.add_argument("--warm-trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--provider", default="external")
    parser.add_argument("--model", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--threads", default="")
    parser.add_argument("--allow-silence-wer", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    trials_path = output_dir / "trials.jsonl"
    events_path.write_text("")
    trials_path.write_text("")

    run_id = time.strftime("%Y%m%dT%H%M%S")
    phrases = list(read_jsonl(Path(args.corpus)))
    if corpus_uses_silence_fallback(phrases) and not args.allow_silence_wer:
        raise SystemExit(
            "corpus contains silence-fallback rows; refusing WER/CER run. "
            "Regenerate with --tts or pass --allow-silence-wer for timing-only experiments."
        )

    for phrase in phrases:
        for trial in range(1, args.warm_trials + 1):
            trial_id = f"{phrase['id']}-{trial}"
            write_jsonl(events_path, {
                "schema": "chezwizper-bench-event-v1",
                "run_id": run_id,
                "trial_id": trial_id,
                "phrase_id": phrase["id"],
                "event": "offline_transcription_begin",
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "extra": {"wav_path": phrase.get("wav_path")},
            })

            text, status, error, elapsed_ms, user_ms, system_ms, peak_rss_mb = run_trial(
                args.command_template,
                phrase,
                args.timeout,
            )
            if phrase.get("tts") == "silence-fallback":
                wer, cer = None, None
            else:
                wer, cer = error_rates(phrase["expected_text"], text)
            audio_ms = phrase.get("duration_ms")
            rtf = round(elapsed_ms / audio_ms, 4) if audio_ms else None

            write_jsonl(events_path, {
                "schema": "chezwizper-bench-event-v1",
                "run_id": run_id,
                "trial_id": trial_id,
                "phrase_id": phrase["id"],
                "event": "offline_transcription_end",
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "extra": {"status": status, "elapsed_ms": elapsed_ms},
            })
            write_jsonl(trials_path, {
                "run_id": run_id,
                "trial_id": trial_id,
                "phrase_id": phrase["id"],
                "phrase_group": phrase["group"],
                "scenario": "offline_transcribe",
                "provider": args.provider,
                "engine_version": None,
                "model": args.model,
                "model_path": args.model_path,
                "threads": args.threads,
                "audio_ms": audio_ms,
                "press_to_recording_start_ms": None,
                "stop_to_wav_ms": None,
                "wav_to_transcription_ms": elapsed_ms,
                "transcription_to_injection_ms": None,
                "total_stop_to_text_ms": elapsed_ms,
                "total_stop_to_text_source": "transcription",
                "rtf": rtf,
                "wer": wer,
                "cer": cer,
                "peak_rss_mb": peak_rss_mb,
                "cpu_user_ms": user_ms,
                "cpu_system_ms": system_ms,
                "text": text,
                "status": status,
                "error": error,
            })

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["summarize.py", "--trials", str(trials_path), "--output-dir", str(output_dir)]
        summarize_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
