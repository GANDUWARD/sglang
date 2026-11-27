"""Fixed-clock CPU value-formula ablation; imports the real cache only when run.

All arms use the same indexed candidates and online learner at learning rate .03.
Only the retention formula changes. This is not GPU latency or output validation.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

FORMULAS = ("full", "no_depth", "no_decay", "no_bonus")
SEEDS = (1729, 1730, 1731, 1732, 1733, 1734)
LEARNING_RATE = 0.03
LEARNER_FIELDS = ("weights", "positive_labels", "negative_labels", "censored_labels")
REQUEST_FIELDS = ("id", "group", "input_tokens", "logical_time_s")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(backend):
    return {
        **backend.source_hashes(),
        "benchmark/prefix_value_term_ablation.py": sha256(Path(__file__)),
    }


def validate_learner_alignment(runs):
    """Require identical observable learning histories, not only final weights."""
    if tuple(run["configuration"] for run in runs) != FORMULAS:
        raise AssertionError("Missing, reordered or duplicate ablation arms")
    baseline = runs[0]
    for run in runs:
        if run["learning_rate"] != LEARNING_RATE or not run["learning_enabled"]:
            raise AssertionError("All value-formula arms must train at learning rate .03")
        if len(run["requests"]) != len(baseline["requests"]):
            raise AssertionError("Arms have different request counts")
        for base_row, row in zip(baseline["requests"], run["requests"]):
            for field in REQUEST_FIELDS + LEARNER_FIELDS:
                if row[field] != base_row[field]:
                    raise AssertionError(
                        f"{run['configuration']} differs in {field} at request {row['id']}"
                    )
        for field in ("pending_labels_at_end", "history_entries"):
            if run[field] != baseline[field]:
                raise AssertionError(f"Learner final state differs in {field}")
    learning_rows = [
        {key: row[key] for key in REQUEST_FIELDS + LEARNER_FIELDS}
        for row in baseline["requests"]
    ]
    return {
        "all_four_arms_identical": True,
        "requests_checked": len(learning_rows),
        "fields_checked": list(REQUEST_FIELDS + LEARNER_FIELDS),
        "learning_history_sha256": hashlib.sha256(
            json.dumps(learning_rows, sort_keys=True).encode()
        ).hexdigest(),
    }


def aggregate_pairs(pairs):
    """Workload-level paired statistics; no request-level significance claim."""
    if not pairs:
        return {"workloads": 0}
    deltas = [pair["reused_tokens_delta"] for pair in pairs]
    relative = [pair["reused_tokens_relative_delta"] for pair in pairs]
    relative = [value for value in relative if value is not None]
    base = sum(pair["baseline_reused_tokens"] for pair in pairs)
    candidate = sum(pair["candidate_reused_tokens"] for pair in pairs)
    return {
        "workloads": len(pairs),
        "win": sum(value > 0 for value in deltas),
        "tie": sum(value == 0 for value in deltas),
        "loss": sum(value < 0 for value in deltas),
        "baseline_reused_tokens_sum": base,
        "candidate_reused_tokens_sum": candidate,
        "pooled_relative_delta": candidate / base - 1 if base else None,
        "per_workload_relative_range": [min(relative), max(relative)] if relative else None,
        "per_workload_absolute_range": [min(deltas), max(deltas)],
    }


def compact_summary(manifest, workloads, pairs):
    keys = ("seed", "split", "capacity_tokens", "interval_s", "configuration")
    one_second = [case for case in workloads if case["interval_s"] == 1.0]
    return {
        "scope": manifest["scope"],
        "provenance": manifest,
        "comparison": "Each removed value term is compared with full, using the same trained learner; this is not a frozen-learner or independent LRU comparison.",
        "statistics_note": "Win/tie/loss uses total reused tokens within paired workloads. Pooled delta is a ratio of sums, not a mean percentage or serving improvement.",
        "split_summaries": {
            split: {
                formula: aggregate_pairs([pair for pair in pairs if pair["split"] == split and pair["configuration"] == formula])
                for formula in FORMULAS[1:]
            }
            for split in ("development", "held_out")
        },
        "per_seed_summaries": {
            str(seed): {
                formula: aggregate_pairs([pair for pair in pairs if pair["seed"] == seed and pair["configuration"] == formula])
                for formula in FORMULAS[1:]
            }
            for seed in manifest["seeds"]
        },
        "workloads": workloads,
        "regressions_vs_full": [pair for pair in pairs if pair["reused_tokens_delta"] < 0],
        "per_group_regressions_vs_full": [
            {**{key: pair[key] for key in keys}, "negative_group_deltas": negative}
            for pair in pairs
            if (negative := {group: delta for group, delta in pair["per_group_reused_tokens_delta"].items() if delta < 0})
        ],
        "one_second_interval": {
            "workloads": len(one_second),
            "all_four_arms_equal": all(len(set(case["reused_tokens"].values())) == 1 for case in one_second) if one_second else None,
        },
        "raw_results_directory": manifest["output_directory"],
        "raw_files": ["manifest.json", "paired_vs_full.json", "repeat_checks.json", "checksums.json", "trace-<seed>.json", "seed-<seed>-cap-<capacity>-gap-<interval>.json"],
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--runtime-image-id", required=True)
    result.add_argument("--cpu-affinity", type=int, nargs="+", help="Optional explicit CPU IDs; effective affinity is always recorded")
    result.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    result.add_argument("--capacities", type=int, nargs="+", default=[2048, 4096])
    result.add_argument("--intervals", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    result.add_argument("--requests", type=int, default=240)
    result.add_argument("--verify-repeat", action="store_true")
    return result


def run_matrix(args, backend):
    """Backend injection permits lightweight orchestration tests without Torch."""
    if args.cpu_affinity is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("Explicit CPU affinity requires sched_setaffinity support")
        os.sched_setaffinity(0, set(args.cpu_affinity))
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if args.cpu_affinity is not None and affinity != sorted(set(args.cpu_affinity)):
        raise AssertionError("Effective CPU affinity differs from the requested mask")
    args.output.mkdir(parents=True, exist_ok=False)
    backend.torch.set_num_threads(1)
    manifest = {
        "status": "running", "started_at_utc": utc_now(),
        "scope": "Real RadixCache CPU semantic value-term ablation; symbolic allocator; no GPU inference, service latency or starvation guarantee",
        "output_directory": str(args.output.resolve()),
        "source_commit": args.source_commit, "source_sha256": source_hashes(backend),
        "runtime_image_id": args.runtime_image_id,
        "python": platform.python_version(), "torch": backend.torch.__version__,
        "requested_cpu_affinity": args.cpu_affinity, "effective_cpu_affinity": affinity,
        "torch_threads": backend.torch.get_num_threads(), "visible_logical_cpus": os.cpu_count(),
        "seeds": args.seeds, "development_seeds": list(SEEDS[:3]), "held_out_seeds": list(SEEDS[3:]),
        "capacities_tokens": args.capacities, "intervals_s": args.intervals,
        "requests_per_arm": args.requests, "formulas": list(FORMULAS),
        "policy": "reuse_value", "learning_rate": LEARNING_RATE,
        "formula_semantics": {
            "full": "Existing probability times external age decay times depth cost proxy, capped bonus plus recency",
            "no_depth": "Remove only the external depth cost multiplier; keep the learner's depth feature",
            "no_decay": "Remove only the external age decay; keep learned reuse-interval/frequency features",
            "no_bonus": "Use recency score; retain indexed candidates and all observation/training work",
        },
        "clock": "request ordinal times fixed interval; reset to zero per arm",
        "reset": "Fresh cache and zero learner weights per arm; no transfer between seeds; no selection on held-out seeds",
        "repeat_validation": args.verify_repeat,
        "tie_note": "Production heap order unchanged; repeat digest excludes CPU timings and exact-score tied victim identity",
    }
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    workloads, pairs, repeats = [], [], []
    try:
        for seed in args.seeds:
            trace = backend.make_trace(seed, args.requests)
            trace_path = args.output / f"trace-{seed}.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            for capacity in args.capacities:
                for interval in args.intervals:
                    workload = {"seed": seed, "split": "development" if seed in SEEDS[:3] else "held_out", "capacity_tokens": capacity, "interval_s": interval, "trace_sha256": sha256(trace_path)}
                    runs = []
                    for formula in FORMULAS:
                        config = (formula, "reuse_value", LEARNING_RATE)
                        run = backend.replay(trace, capacity, interval, config, prefix_value_formula=formula)
                        if run["cache_stats"].get("value_formula") != formula:
                            raise AssertionError("Effective production value formula differs from requested arm")
                        runs.append(run)
                        if args.verify_repeat:
                            repeat = backend.replay(trace, capacity, interval, config, prefix_value_formula=formula)
                            matches = repeat["semantic_sha256"] == run["semantic_sha256"]
                            repeats.append({**workload, "formula": formula, "matches": matches, "first_sha256": run["semantic_sha256"], "repeat_sha256": repeat["semantic_sha256"]})
                            if not matches:
                                write_json(args.output / "repeat_failure.json", {"workload": workload, "first": run, "repeat": repeat})
                                raise AssertionError("Fixed-clock semantic repeat differs")
                    raw_path = args.output / f"seed-{seed}-cap-{capacity}-gap-{interval:g}.json"
                    write_json(raw_path, {"workload": workload, "runs": runs})
                    alignment = validate_learner_alignment(runs)
                    case_pairs = [{**workload, **backend.paired_summary(runs[0], run)} for run in runs[1:]]
                    pairs.extend(case_pairs)
                    workloads.append({**workload, "raw_file": raw_path.name, "learner_alignment": alignment, "reused_tokens": {run["configuration"]: run["reused_tokens"] for run in runs}, "delta_vs_full": {pair["configuration"]: pair["reused_tokens_delta"] for pair in case_pairs}})
                    print(json.dumps(workloads[-1]), flush=True)
        if source_hashes(backend) != manifest["source_sha256"]:
            raise AssertionError("Source files changed during the matrix")
        manifest.update(status="complete", finished_at_utc=utc_now(), workloads=len(workloads), measured_runs=len(workloads) * 4, measured_requests=len(workloads) * 4 * args.requests, repeat_runs=len(repeats), all_learner_histories_match=True)
        write_json(args.output / "paired_vs_full.json", pairs)
        write_json(args.output / "repeat_checks.json", repeats)
        write_json(args.output / "summary_compact.json", compact_summary(manifest, workloads, pairs))
    except Exception as error:
        manifest.update(status="failed", finished_at_utc=utc_now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(manifest_path, manifest)
    write_json(args.output / "checksums.json", {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()})


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.requests < 1 or any(not math.isfinite(value) or value <= 0 for value in args.capacities + args.intervals):
        cli.error("Requests, capacities and intervals must be finite and positive")
    if any(seed not in SEEDS for seed in args.seeds):
        cli.error("Use the predefined development or held-out seeds")
    if any(len(set(values)) != len(values) for values in (args.seeds, args.capacities, args.intervals)):
        cli.error("Matrix dimensions must not contain duplicates")
    if args.cpu_affinity is not None and (any(cpu < 0 for cpu in args.cpu_affinity) or len(set(args.cpu_affinity)) != len(args.cpu_affinity)):
        cli.error("CPU IDs must be unique and non-negative")
    # --help and argument validation above intentionally avoid heavy imports.
    import prefix_value_learner_replay as backend

    run_matrix(args, backend)


if __name__ == "__main__":
    main(sys.argv[1:])
