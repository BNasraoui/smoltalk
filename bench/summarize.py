#!/usr/bin/env python3
"""Summarize benchmark trials into summary.csv and baseline_report.md."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


TIMING_COLUMNS = [
    "stop_to_wav_ms",
    "wav_to_transcription_ms",
    "transcription_to_injection_ms",
    "total_stop_to_text_ms",
    "rtf",
    "peak_rss_mb",
    "cpu_user_ms",
    "cpu_system_ms",
]


CSV_FIELDNAMES = [
    "run_id", "provider", "engine_version", "model", "model_path", "threads", "scenario", "n",
    "phrase_group", "text_source", "audio_ms_mean", "press_to_recording_start_p50_ms",
    "press_to_recording_start_p95_ms", "stop_to_wav_p50_ms",
    "wav_to_transcription_p50_ms", "transcription_to_injection_p50_ms",
    "total_stop_to_text_p50_ms", "total_stop_to_text_p95_ms", "rtf_p50",
    "wer_mean", "cer_mean", "peak_rss_mb", "cpu_user_ms_mean", "cpu_system_ms_mean", "failures",
]


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


def percentile(values, pct):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    index = round((len(values) - 1) * pct)
    return values[index]


def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.fmean(values), 3) if values else None


def common_text_source(rows):
    sources = sorted({r.get("total_stop_to_text_source") for r in rows if r.get("total_stop_to_text_source")})
    if not sources:
        return ""
    return sources[0] if len(sources) == 1 else "mixed"


def summarize(trials):
    groups = defaultdict(list)
    for row in trials:
        key = (
            row.get("run_id"),
            row.get("provider"),
            row.get("model"),
            row.get("scenario"),
            row.get("phrase_group"),
        )
        groups[key].append(row)

    output = []
    for (run_id, provider, model, scenario, phrase_group), rows in sorted(groups.items()):
        summary = {
            "run_id": run_id,
            "provider": provider,
            "engine_version": rows[0].get("engine_version"),
            "model": model,
            "model_path": rows[0].get("model_path"),
            "threads": rows[0].get("threads"),
            "scenario": scenario,
            "n": len(rows),
            "phrase_group": phrase_group,
            "text_source": common_text_source(rows),
            "audio_ms_mean": mean([r.get("audio_ms") for r in rows]),
            "press_to_recording_start_p50_ms": percentile([r.get("press_to_recording_start_ms") for r in rows], 0.50),
            "press_to_recording_start_p95_ms": percentile([r.get("press_to_recording_start_ms") for r in rows], 0.95),
            "stop_to_wav_p50_ms": percentile([r.get("stop_to_wav_ms") for r in rows], 0.50),
            "wav_to_transcription_p50_ms": percentile([r.get("wav_to_transcription_ms") for r in rows], 0.50),
            "transcription_to_injection_p50_ms": percentile([r.get("transcription_to_injection_ms") for r in rows], 0.50),
            "total_stop_to_text_p50_ms": percentile([r.get("total_stop_to_text_ms") for r in rows], 0.50),
            "total_stop_to_text_p95_ms": percentile([r.get("total_stop_to_text_ms") for r in rows], 0.95),
            "rtf_p50": percentile([r.get("rtf") for r in rows], 0.50),
            "wer_mean": mean([r.get("wer") for r in rows]),
            "cer_mean": mean([r.get("cer") for r in rows]),
            "peak_rss_mb": percentile([r.get("peak_rss_mb") for r in rows], 0.95),
            "cpu_user_ms_mean": mean([r.get("cpu_user_ms") for r in rows]),
            "cpu_system_ms_mean": mean([r.get("cpu_system_ms") for r in rows]),
            "failures": sum(1 for r in rows if r.get("status") != "ok"),
        }
        output.append(summary)
    return output


def write_report(path, summary_rows, trial_count):
    injection_measured = any(row.get("text_source") == "injection" for row in summary_rows)
    lines = [
        "# ChezWizper Baseline Benchmark Report",
        "",
        f"Trials: {trial_count}",
        "",
        "Injection was measured." if injection_measured else "Injection was not measured; stop-to-text ends at clipboard or idle.",
        "",
        "| scenario | group | text source | n | stop-to-text p50 ms | stop-to-text p95 ms | failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('scenario')} | {row.get('phrase_group')} | {row.get('text_source')} | {row.get('n')} | "
            f"{row.get('total_stop_to_text_p50_ms')} | {row.get('total_stop_to_text_p95_ms')} | {row.get('failures')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default="bench-artifacts/trials.jsonl")
    parser.add_argument("--output-dir", default="bench-artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = read_jsonl(Path(args.trials))
    summary_rows = summarize(trials)

    csv_path = output_dir / "summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else CSV_FIELDNAMES
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_report(output_dir / "baseline_report.md", summary_rows, len(trials))
    print(f"wrote {csv_path} and {output_dir / 'baseline_report.md'}")


if __name__ == "__main__":
    main()
