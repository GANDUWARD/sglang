"""Lightweight orchestration checks; no Torch, GPU, cache replay or measurements."""

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "value_term_ablation_under_test", ROOT / "benchmark/prefix_value_term_ablation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example_run(formula, count=2):
    reuse = {"full": 1, "no_depth": 0, "no_decay": 2, "no_bonus": 1}[formula]
    rows = [
        {
            "id": index,
            "group": 0,
            "input_tokens": 2,
            "logical_time_s": (index + 1) * 0.1,
            "weights": [0.1 * index, 0.0, 0.0, 0.0],
            "positive_labels": index,
            "negative_labels": 0,
            "censored_labels": 0,
            "reused_tokens": reuse,
        }
        for index in range(count)
    ]
    return {
        "configuration": formula,
        "learning_rate": 0.03,
        "learning_enabled": True,
        "requests": rows,
        "reused_tokens": reuse * count,
        "pending_labels_at_end": 1,
        "history_entries": 1,
        "cache_stats": {"value_formula": formula},
        "semantic_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
    }


def fake_backend(calls, divergent=False):
    def replay(trace, capacity, interval, config, *, prefix_value_formula):
        calls.append((capacity, interval, config, prefix_value_formula))
        run = example_run(prefix_value_formula, len(trace))
        if divergent and prefix_value_formula == "no_decay":
            run["requests"][0]["negative_labels"] += 1
        return run

    def pair(base, candidate):
        delta = candidate["reused_tokens"] - base["reused_tokens"]
        return {
            "baseline": base["configuration"],
            "configuration": candidate["configuration"],
            "baseline_reused_tokens": base["reused_tokens"],
            "candidate_reused_tokens": candidate["reused_tokens"],
            "reused_tokens_delta": delta,
            "reused_tokens_relative_delta": delta / base["reused_tokens"],
            "per_group_reused_tokens_delta": {"0": delta},
        }

    return types.SimpleNamespace(
        source_hashes=lambda: {"mock_backend": "not-runtime-evidence"},
        torch=types.SimpleNamespace(
            set_num_threads=lambda count: None,
            get_num_threads=lambda: 1,
            __version__="mock-not-torch",
        ),
        make_trace=lambda seed, count: [
            {"id": index, "group": 0, "input_ids": [1, 2]}
            for index in range(count)
        ],
        replay=replay,
        paired_summary=pair,
    )


class TestValueTermAblation(unittest.TestCase):
    def test_help_avoids_real_backend_import(self):
        self.assertNotIn("prefix_value_learner_replay", sys.modules)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            MODULE.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertNotIn("prefix_value_learner_replay", sys.modules)

    def test_alignment_checks_all_requests_not_only_final_weights(self):
        runs = [example_run(formula) for formula in MODULE.FORMULAS]
        self.assertTrue(MODULE.validate_learner_alignment(runs)["all_four_arms_identical"])
        for field in MODULE.LEARNER_FIELDS + MODULE.REQUEST_FIELDS:
            with self.subTest(field=field):
                changed = copy.deepcopy(runs)
                row = changed[1]["requests"][0]
                row[field] = [99.0] * 4 if field == "weights" else 99
                with self.assertRaisesRegex(AssertionError, field):
                    MODULE.validate_learner_alignment(changed)

    def test_no_bonus_must_still_train(self):
        runs = [example_run(formula) for formula in MODULE.FORMULAS]
        runs[-1]["learning_enabled"] = False
        with self.assertRaisesRegex(AssertionError, "must train"):
            MODULE.validate_learner_alignment(runs)

    def test_workload_regressions_are_retained(self):
        pairs = [
            {
                "baseline_reused_tokens": 100,
                "candidate_reused_tokens": value,
                "reused_tokens_delta": value - 100,
                "reused_tokens_relative_delta": value / 100 - 1,
            }
            for value in (120, 100, 90)
        ]
        stats = MODULE.aggregate_pairs(pairs)
        self.assertEqual((stats["win"], stats["tie"], stats["loss"]), (1, 1, 1))
        self.assertEqual(stats["per_workload_absolute_range"], [-10, 20])
        self.assertAlmostEqual(stats["pooled_relative_delta"], 310 / 300 - 1)

    def test_mock_orchestration_writes_raw_compact_and_checksums(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / "mock-output"
            args = MODULE.parser().parse_args([
                "--output", str(output), "--source-commit", "mock",
                "--runtime-image-id", "mock", "--seeds", "1729",
                "--capacities", "2048", "--intervals", "0.1",
                "--requests", "2", "--verify-repeat",
            ])
            calls = []
            with redirect_stdout(io.StringIO()):
                MODULE.run_matrix(args, fake_backend(calls))
            self.assertEqual(len(calls), 8)
            self.assertTrue(all(call[2] == (call[3], "reuse_value", 0.03) for call in calls))
            compact = json.loads((output / "summary_compact.json").read_text())
            self.assertEqual(compact["provenance"]["status"], "complete")
            self.assertTrue(compact["provenance"]["all_learner_histories_match"])
            self.assertEqual(compact["regressions_vs_full"][0]["configuration"], "no_depth")
            self.assertEqual(compact["per_group_regressions_vs_full"][0]["negative_group_deltas"], {"0": -2})
            self.assertIsNone(compact["one_second_interval"]["all_four_arms_equal"])
            checksums = json.loads((output / "checksums.json").read_text())
            self.assertTrue(all(MODULE.sha256(output / name) == digest for name, digest in checksums.items()))

    def test_divergent_learner_fails_and_preserves_diagnostic_raw(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory) / "mock-output"
            args = MODULE.parser().parse_args([
                "--output", str(output), "--source-commit", "mock",
                "--runtime-image-id", "mock", "--seeds", "1729",
                "--capacities", "2048", "--intervals", "0.1", "--requests", "2",
            ])
            with self.assertRaisesRegex(AssertionError, "negative_labels"):
                MODULE.run_matrix(args, fake_backend([], divergent=True))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((output / "seed-1729-cap-2048-gap-0.1.json").exists())
            self.assertFalse((output / "summary_compact.json").exists())


if __name__ == "__main__":
    unittest.main()
