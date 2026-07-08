#!/usr/bin/env python3
"""Focused regression tests for the benchmark harness helpers."""

import json
import tempfile
import time
import unittest
from pathlib import Path

import bench_e2e
import make_corpus
import summarize


class BenchmarkHarnessTests(unittest.TestCase):
    def test_generated_config_preserves_base_and_applies_safe_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "config.toml"
            generated = Path(tmp) / "bench-config.toml"
            base.write_text(
                "[whisper]\nmodel = \"base.en\"\nthreads = 4\n"
                "[behavior]\nauto_paste = true\naudio_feedback = true\n"
                "[ui]\nshow_notifications = true\n"
                "[api]\nport = 3737\n"
            )

            config = bench_e2e.DaemonConfig.from_base(base, generated, port=3838, measure_injection=False)

            self.assertEqual(config.api_url, "http://127.0.0.1:3838")
            written = bench_e2e.load_toml(generated)
            self.assertEqual(written["whisper"]["model"], "base.en")
            self.assertEqual(written["whisper"]["threads"], 4)
            self.assertEqual(written["api"]["port"], 3838)
            self.assertFalse(written["behavior"]["auto_paste"])
            self.assertFalse(written["behavior"]["audio_feedback"])
            self.assertFalse(written["ui"]["show_notifications"])
            self.assertTrue(written["behavior"]["delete_audio_files"])

    def test_incremental_trace_reader_tolerates_only_final_malformed_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"event":"ok"}\n{"event":')
            reader = bench_e2e.IncrementalJsonlReader(path)

            self.assertEqual([r["event"] for r in reader.read_new()], ["ok"])

            with path.open("a") as handle:
                handle.write(' "complete"}\n{bad}\n{"event":"later"}\n')
            with self.assertRaisesRegex(ValueError, "malformed JSONL"):
                reader.read_new()

    def test_trial_row_uses_samples_for_audio_ms_rtf_and_text_source(self):
        events = [
            {"event": "audio_stop_begin", "monotonic_ns": 1_000_000_000, "extra": {}},
            {"event": "samples_taken", "monotonic_ns": 1_001_000_000, "extra": {"samples": 24_000}},
            {"event": "transcription_begin", "monotonic_ns": 1_100_000_000, "extra": {}},
            {"event": "transcription_end", "monotonic_ns": 1_850_000_000, "extra": {}},
            {"event": "clipboard_copy_end", "monotonic_ns": 1_900_000_000, "extra": {}},
            {"event": "state_idle_set", "monotonic_ns": 1_950_000_000, "extra": {}},
        ]

        row = bench_e2e.build_trial_row(
            run_id="run",
            trial_id="trial",
            phrase={"id": "phrase", "group": "short", "duration_ms": 9999},
            trial_events=events,
            provider="chezwizper",
            model="base",
            model_path="",
            threads="",
            resources=bench_e2e.ResourceStats(peak_rss_mb=123.0, cpu_user_ms=40.0, cpu_system_ms=5.0),
        )

        self.assertEqual(row["audio_ms"], 1500.0)
        self.assertEqual(row["rtf"], 0.5)
        self.assertEqual(row["total_stop_to_text_source"], "clipboard")
        self.assertEqual(row["total_stop_to_text_ms"], 900.0)

    def test_proc_snapshot_diffs_process_tree_cpu_and_peak_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            (proc / "100" / "task" / "100").mkdir(parents=True)
            (proc / "101" / "task" / "101").mkdir(parents=True)
            (proc / "100" / "task" / "100" / "children").write_text("101\n")
            (proc / "101" / "task" / "101" / "children").write_text("")
            (proc / "100" / "status").write_text("VmRSS:\t1024 kB\nVmHWM:\t2048 kB\n")
            (proc / "101" / "status").write_text("VmRSS:\t4096 kB\nVmHWM:\t8192 kB\n")
            stat = " ".join(["100", "(cmd)", "S", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "10", "5"])
            child_stat = " ".join(["101", "(cmd)", "S", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "20", "10"])
            (proc / "100" / "stat").write_text(stat)
            (proc / "101" / "stat").write_text(child_stat)

            sampler = bench_e2e.ProcTreeSampler(pid=100, proc_root=proc, clock_ticks=100)
            first = sampler.snapshot()
            (proc / "100" / "stat").write_text(stat.replace("10 5", "20 7"))
            (proc / "101" / "stat").write_text(child_stat.replace("20 10", "25 15"))
            second = sampler.snapshot()

            stats = bench_e2e.ResourceStats.from_snapshots(first, second)
            self.assertEqual(stats.peak_rss_mb, 10.0)
            self.assertEqual(stats.cpu_user_ms, 150.0)
            self.assertEqual(stats.cpu_system_ms, 70.0)

    def test_flite_command_and_silence_detection(self):
        row = {"id": "p1", "spoken_prompt": "hello world"}
        command = make_corpus.tts_command("flite", row["spoken_prompt"], Path("out.wav"))
        self.assertEqual(command, ["flite", "-t", "hello world", "-o", "out.wav"])
        self.assertTrue(make_corpus.corpus_uses_silence_fallback([{"tts": "silence-fallback"}]))

    def test_summary_includes_text_source_and_injection_note(self):
        rows = summarize.summarize([
            {
                "run_id": "r",
                "provider": "p",
                "model": "m",
                "scenario": "e2e",
                "phrase_group": "short",
                "total_stop_to_text_source": "clipboard",
                "status": "ok",
            }
        ])

        self.assertEqual(rows[0]["text_source"], "clipboard")
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            summarize.write_report(report, rows, 1)
            self.assertIn("Injection was not measured", report.read_text())


if __name__ == "__main__":
    unittest.main()
