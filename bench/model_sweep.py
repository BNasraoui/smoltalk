#!/usr/bin/env python3
"""Run the offline whisper.cpp model sweep for ChezWizper."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_WHISPER_CLI = Path("/home/ben/.local/share/chezwizper/whisper/build/bin/whisper-cli")
DEFAULT_MODEL_DIR = Path("/home/ben/.local/share/chezwizper/whisper/models")
DEFAULT_CORPUS = Path("bench-artifacts/corpus/corpus.jsonl")
DEFAULT_OUTPUT_ROOT = Path("bench-artifacts/model-sweep")

MODELS = {
    "tiny.en": "ggml-tiny.en.bin",
    "tiny.en-q5_1": "ggml-tiny.en-q5_1.bin",
    "base.en": "ggml-base.en.bin",
    "base.en-q5_0": "ggml-base.en-q5_0.bin",
    "base.en-q5_1": "ggml-base.en-q5_1.bin",
    "small.en-q5_1": "ggml-small.en-q5_1.bin",
    "large-v3-turbo-q5_1": "ggml-large-v3-turbo-q5_1.bin",
}

DEFAULT_CANDIDATES = (
    "tiny.en",
    "tiny.en-q5_1",
    "base.en",
    "base.en-q5_0",
    "base.en-q5_1",
    "small.en-q5_1",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-cli", type=Path, default=DEFAULT_WHISPER_CLI)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--include-large-anchor", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--anchor-trials", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--limit", type=int, help="Use only the first N corpus rows.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--force", action="store_true", help="Remove an existing run directory before running.")
    parser.add_argument("--extra-name", default="", help="Suffix output directories, for thread/beam variants.")
    return parser.parse_args()


def load_jsonl(path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def limited_corpus(source, output_root, limit):
    if not limit:
        return source
    rows = list(load_jsonl(source))[:limit]
    if not rows:
        raise SystemExit(f"no corpus rows found in {source}")
    path = output_root / f"corpus-limit-{limit}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def model_path(model_dir, model_name):
    try:
        filename = MODELS[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODELS))
        raise SystemExit(f"unknown model {model_name!r}; valid models: {valid}") from exc
    path = model_dir / filename
    if not path.exists():
        raise SystemExit(f"missing model file: {path}")
    return path


def output_name(model_name, threads, beam_size, extra_name):
    parts = [model_name, f"t{threads}"]
    if beam_size is not None:
        parts.append(f"bs{beam_size}")
    if extra_name:
        parts.append(extra_name)
    return "_".join(parts)


def run_one(args, model_name, corpus, trials):
    model = model_path(args.model_dir, model_name)
    out_dir = args.output_root / output_name(model_name, args.threads, args.beam_size, args.extra_name)
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} already exists; pass --force to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    command = [
        str(args.whisper_cli),
        "-m",
        str(model),
        "-f",
        "{wav}",
        "-l",
        "en",
        "-nt",
        "-np",
        "-t",
        str(args.threads),
    ]
    if args.beam_size is not None:
        command.extend(["-bs", str(args.beam_size)])

    bench_cmd = [
        sys.executable,
        "bench/bench_transcribe.py",
        "--corpus",
        str(corpus),
        "--output-dir",
        str(out_dir),
        "--command-template",
        " ".join(command),
        "--warm-trials",
        str(trials),
        "--timeout",
        str(args.timeout),
        "--provider",
        "whisper-cli",
        "--model",
        model_name,
        "--model-path",
        str(model),
        "--threads",
        str(args.threads),
    ]

    print(f"running {model_name}: trials={trials} threads={args.threads} output={out_dir}", flush=True)
    subprocess.run(bench_cmd, check=True)


def main():
    args = parse_args()
    if not args.whisper_cli.exists():
        raise SystemExit(f"missing whisper-cli binary: {args.whisper_cli}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    corpus = limited_corpus(args.corpus, args.output_root, args.limit)

    models = list(args.models)
    if args.include_large_anchor and "large-v3-turbo-q5_1" not in models:
        models.append("large-v3-turbo-q5_1")

    for model_name in models:
        trials = args.anchor_trials if model_name == "large-v3-turbo-q5_1" else args.trials
        run_one(args, model_name, corpus, trials)


if __name__ == "__main__":
    main()
