#!/usr/bin/env python3
"""Materialize the benchmark phrase corpus and optional WAV fixtures."""

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


TARGET_RATE = 16000
TARGET_CHANNELS = 1
TARGET_WIDTH = 2


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_silence_wav(path, seconds=1.2, sample_rate=TARGET_RATE):
    frames = int(seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def tts_command(engine, text, wav_path):
    if engine == "flite":
        return ["flite", "-t", text, "-o", str(wav_path)]
    if engine == "espeak-ng":
        return ["espeak-ng", "-w", str(wav_path), "-s", "150", text]
    if engine == "espeak":
        return ["espeak", "-w", str(wav_path), "-s", "150", text]
    raise ValueError(f"unsupported TTS engine: {engine}")


def choose_tts_engine(requested):
    engines = ["flite", "espeak-ng", "espeak"] if requested == "auto" else [requested]
    for engine in engines:
        if shutil.which(engine):
            return engine
    return None


def synthesize(engine, text, wav_path):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            tts_command(engine, text, Path(tmp.name)),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        normalize_wav(Path(tmp.name), wav_path)


def normalize_wav(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(input_path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = decode_pcm(raw, width, channels)
    if rate != TARGET_RATE:
        samples = resample_nearest(samples, rate, TARGET_RATE)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_WIDTH)
        wav.setframerate(TARGET_RATE)
        wav.writeframes(encode_pcm16(samples))


def decode_pcm(raw, width, channels):
    if channels < 1:
        raise ValueError("WAV must have at least one channel")
    frame_width = width * channels
    samples = []
    for index in range(0, len(raw), frame_width):
        channel_values = []
        for channel in range(channels):
            start = index + channel * width
            sample = raw[start : start + width]
            if width == 1:
                value = (sample[0] - 128) << 8
            elif width == 2:
                value = int.from_bytes(sample, "little", signed=True)
            elif width == 4:
                value = int.from_bytes(sample, "little", signed=True) >> 16
            else:
                raise ValueError(f"unsupported WAV sample width: {width}")
            channel_values.append(value)
        samples.append(int(sum(channel_values) / len(channel_values)))
    return samples


def resample_nearest(samples, source_rate, target_rate):
    if not samples or source_rate == target_rate:
        return samples
    target_len = max(1, round(len(samples) * target_rate / source_rate))
    return [samples[min(len(samples) - 1, int(i * source_rate / target_rate))] for i in range(target_len)]


def encode_pcm16(samples):
    out = bytearray()
    for sample in samples:
        clipped = max(-32768, min(32767, int(sample)))
        out.extend(clipped.to_bytes(2, "little", signed=True))
    return bytes(out)


def wav_duration_ms(path):
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() * 1000 / wav.getframerate())


def corpus_uses_silence_fallback(rows):
    return any(row.get("tts") == "silence-fallback" for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="bench/corpus.jsonl")
    parser.add_argument("--output-dir", default="bench-artifacts/corpus")
    parser.add_argument("--tts", nargs="?", const="auto", default=None, help="Use flite/espeak when available")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    engine = choose_tts_engine(args.tts or "none") if args.tts else None

    for row in read_jsonl(Path(args.input)):
        wav_path = wav_dir / f"{row['id']}.wav"
        tts = "silence-fallback"
        if engine:
            synthesize(engine, row["spoken_prompt"], wav_path)
            tts = engine
        else:
            write_silence_wav(wav_path)

        row = dict(row)
        row["wav_path"] = str(wav_path)
        row["duration_ms"] = wav_duration_ms(wav_path)
        row["tts"] = tts
        rows.append(row)

    with (output_dir / "corpus.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"wrote {len(rows)} phrases to {output_dir / 'corpus.jsonl'}")
    if corpus_uses_silence_fallback(rows):
        print("warning: corpus contains silence-fallback rows; WER/CER are not meaningful")


if __name__ == "__main__":
    main()
