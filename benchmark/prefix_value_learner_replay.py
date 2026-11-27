"""Paired fixed-clock learning-rate replay through the real CPU RadixCache.

This is a cache semantic experiment, not GPU serving or model-output validation.
It reuses the explicit unavailable-HostKVCache shim from the existing CPU replay.
No production algorithm or learner default is changed. All configurations start
with a fresh cache and zero weights; held-out seeds are reported without tuning.
"""

import argparse
import hashlib
import json
import math
import platform
import time
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from prefix_value_cpu_replay import (
    CacheInitParams,
    RadixCache,
    RadixKey,
    radix_module,
    torch,
)
from prefix_value_replay import make_trace


CONFIGURATIONS = (
    ("frozen", "reuse_value_frozen", 0.03),
    ("lr_0.01", "reuse_value", 0.01),
    ("lr_0.03", "reuse_value", 0.03),
    ("lr_0.10", "reuse_value", 0.10),
)
DEVELOPMENT_SEEDS = (1729, 1730, 1731)
HELD_OUT_SEEDS = (1732, 1733, 1734)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = (
        "benchmark/prefix_value_learner_replay.py",
        "benchmark/prefix_value_cpu_replay.py",
        "benchmark/prefix_value_replay.py",
        "python/sglang/srt/mem_cache/radix_cache.py",
        "python/sglang/srt/mem_cache/reuse_value.py",
        "python/sglang/srt/mem_cache/cache_init_params.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
    }


def check_cache(cache, capacity):
    """Check tree/index/accounting after every completed logical request."""
    stack = list(cache.root_node.children.values())
    leaves = set()
    total = 0
    while stack:
        node = stack.pop()
        if node.lock_ref != 0:
            raise AssertionError("Completed request left a non-root lock")
        if len(node.key) != len(node.value):
            raise AssertionError("Node key/value length mismatch")
        total += len(node.value)
        if not node.children:
            leaves.add(node)
        for child in node.children.values():
            if child.parent is not node:
                raise AssertionError("Broken parent link")
            stack.append(child)
    if leaves != cache._leaf_index:
        raise AssertionError("Structural leaf index differs from actual tree")
    if total != cache.total_size() or not 0 <= total <= capacity:
        raise AssertionError("Cache capacity/accounting invariant violated")
    if cache.protected_size() != 0 or cache.evictable_size() != total:
        raise AssertionError("Cache lock accounting invariant violated")
    return total


def replay(trace, capacity, interval, configuration, *, prefix_value_formula=None):
    name, policy, learning_rate = configuration
    if not trace or max(len(item["input_ids"]) for item in trace) > capacity:
        raise ValueError("Each request must fit in the non-empty replay cache")
    clock = [0.0]
    previous_time = radix_module.time
    radix_module.time = types.SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter
    )
    freed = []
    allocator = Mock()
    allocator.device = torch.device("cpu")

    def record_free(values):
        # These are symbolic token-valued CPU tensors, not physical GPU slots.
        freed.append(
            {"tokens": len(values), "symbolic_values_sha256": digest_json(values.tolist())}
        )

    allocator.free.side_effect = record_free
    started = time.perf_counter()
    try:
        extra_params = (
            {"prefix_value_formula": prefix_value_formula}
            if prefix_value_formula is not None
            else {}
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                eviction_policy=policy,
                **extra_params,
            )
        )
        # RadixCache construction calls reset(), which creates its learner.
        model = cache.reuse_model
        model.learning_rate = learning_rate
        if model.weights != [0.0] * 4 or model.tick != 0:
            raise AssertionError("Replay must begin with a reset learner")
        rows = []
        group_totals = defaultdict(lambda: {"requests": 0, "input_tokens": 0, "reused_tokens": 0})
        for ordinal, item in enumerate(trace, 1):
            # Multiplication avoids wall-clock effects and cumulative clock drift.
            clock[0] = ordinal * interval
            tokens = item["input_ids"]
            free_offset = len(freed)
            operation_start = time.perf_counter()
            cache.observe_request(
                types.SimpleNamespace(origin_input_ids=tokens, extra_key=None)
            )
            match = cache.match_prefix(RadixKey(tokens))
            reused = len(match.device_indices)
            if match.device_indices.tolist() != tokens[:reused]:
                raise AssertionError("Returned cached prefix differs from request")
            cache.inc_lock_ref(match.last_device_node)
            size_before = cache.total_size()
            required = max(0, size_before + len(tokens) - reused - capacity)
            if required:
                cache.evict(required)
            cache.insert(RadixKey(tokens), torch.tensor(tokens))
            cache.dec_lock_ref(match.last_device_node)
            operation_seconds = time.perf_counter() - operation_start
            size_after = check_cache(cache, capacity)
            events = freed[free_offset:]
            evicted = sum(event["tokens"] for event in events)
            if evicted < required or size_after != size_before + len(tokens) - reused - evicted:
                raise AssertionError("Eviction did not satisfy allocation accounting")
            if not all(math.isfinite(weight) for weight in model.weights):
                raise AssertionError("Learner produced non-finite weights")
            row = {
                "id": item["id"],
                "group": item["group"],
                "phase": 0 if ordinal <= len(trace) // 2 else 1,
                "logical_time_s": clock[0],
                "input_tokens": len(tokens),
                "reused_tokens": reused,
                "recomputed_tokens": len(tokens) - reused,
                "required_eviction_tokens": required,
                "evicted_tokens": evicted,
                "eviction_events": events,
                "resident_tokens": size_after,
                "positive_labels": model.positive_labels,
                "negative_labels": model.negative_labels,
                "censored_labels": model.censored_labels,
                "weights": list(model.weights),
                "cpu_operation_s": operation_seconds,
            }
            rows.append(row)
            group = group_totals[str(item["group"])]
            group["requests"] += 1
            group["input_tokens"] += len(tokens)
            group["reused_tokens"] += reused
        if name == "frozen" and model.weights != [0.0] * 4:
            raise AssertionError("Frozen control updated its weights")
        if model.tick != len(trace) or cache.get_research_stats()["observations"] != len(trace):
            raise AssertionError("Logical requests were not observed exactly once")
        # Victim identity among exact-score ties is not a serving-correctness claim.
        # Repeat validation covers reuse, accounting and learning, not CPU timing.
        semantic_rows = [
            {key: value for key, value in row.items() if key not in ("cpu_operation_s", "eviction_events")}
            for row in rows
        ]
        return {
            "configuration": name,
            "policy": policy,
            "learning_rate": learning_rate,
            "learning_enabled": model.learn,
            "request_count": len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "reused_tokens": sum(row["reused_tokens"] for row in rows),
            "recomputed_tokens": sum(row["recomputed_tokens"] for row in rows),
            "evicted_tokens": sum(row["evicted_tokens"] for row in rows),
            "cpu_replay_wall_s": time.perf_counter() - started,
            "cpu_operation_s": sum(row["cpu_operation_s"] for row in rows),
            "per_group": dict(group_totals),
            "semantic_sha256": digest_json(semantic_rows),
            "cache_stats": cache.get_research_stats(),
            "pending_labels_at_end": len(model.pending),
            "history_entries": len(model.records),
            "requests": rows,
        }
    finally:
        radix_module.time = previous_time


def paired_summary(baseline, candidate):
    if [row["id"] for row in baseline["requests"]] != [row["id"] for row in candidate["requests"]]:
        raise AssertionError("Paired runs have different request identities")
    base = baseline["reused_tokens"]
    differences = [
        candidate_row["reused_tokens"] - baseline_row["reused_tokens"]
        for baseline_row, candidate_row in zip(baseline["requests"], candidate["requests"])
    ]
    return {
        "baseline": baseline["configuration"],
        "configuration": candidate["configuration"],
        "learning_rate": candidate["learning_rate"],
        "baseline_reused_tokens": base,
        "candidate_reused_tokens": candidate["reused_tokens"],
        "reused_tokens_delta": sum(differences),
        "reused_tokens_relative_delta": candidate["reused_tokens"] / base - 1 if base else None,
        "requests_with_more_reuse": sum(value > 0 for value in differences),
        "requests_with_equal_reuse": sum(value == 0 for value in differences),
        "requests_with_less_reuse": sum(value < 0 for value in differences),
        "per_group_reused_tokens_delta": {
            group: candidate["per_group"][group]["reused_tokens"] - stats["reused_tokens"]
            for group, stats in baseline["per_group"].items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEVELOPMENT_SEEDS + HELD_OUT_SEEDS))
    parser.add_argument("--capacities", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--intervals", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--requests", type=int, default=240)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--verify-repeat", action="store_true")
    args = parser.parse_args(argv)
    if args.requests < 1 or any(value <= 0 for value in args.capacities + args.intervals):
        parser.error("requests, capacities and intervals must be positive")
    if any(seed not in DEVELOPMENT_SEEDS + HELD_OUT_SEEDS for seed in args.seeds):
        parser.error("Use the predefined development or held-out seeds")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.capacities)) != len(args.capacities) or len(set(args.intervals)) != len(args.intervals):
        parser.error("Matrix dimensions must not contain duplicate values")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = {
        "status": "running",
        "started_at_utc": utc_now(),
        "scope": "Real RadixCache CPU semantic replay; symbolic allocator; no GPU inference or serving latency",
        "source_commit": args.source_commit,
        "source_sha256": source_hashes(),
        "runtime_image_id": args.runtime_image_id,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "executed_seeds": args.seeds,
        "capacities": args.capacities,
        "intervals_s": args.intervals,
        "requests_per_run": args.requests,
        "configurations": [dict(zip(("name", "policy", "learning_rate"), item)) for item in CONFIGURATIONS],
        "repeat_validation": args.verify_repeat,
        "clock": "request ordinal multiplied by fixed interval; starts at zero for each configuration",
        "state_reset": "New cache and zero weights for every workload/configuration; no state transfer between seeds",
        "held_out_protocol": "All preregistered configurations reported; no selection or tuning on held-out results",
        "tie_note": "Production heap/candidate order is unchanged; repeat digest excludes CPU timing and exact-score victim identity",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "started", "utc": manifest["started_at_utc"]}), flush=True)
    summaries = []
    pairs = []
    repeat_checks = []
    for seed in args.seeds:
        trace = make_trace(seed, args.requests)
        trace_payload = json.dumps(trace)
        trace_path = args.output / f"trace-{seed}.json"
        trace_path.write_text(trace_payload)
        trace_sha256 = hashlib.sha256(trace_payload.encode()).hexdigest()
        for capacity in args.capacities:
            for interval in args.intervals:
                workload = {
                    "seed": seed,
                    "split": "development" if seed in DEVELOPMENT_SEEDS else "held_out",
                    "capacity_tokens": capacity,
                    "interval_s": interval,
                    "trace_sha256": trace_sha256,
                }
                runs = []
                for config in CONFIGURATIONS:
                    result = replay(trace, capacity, interval, config)
                    if args.verify_repeat:
                        repeat = replay(trace, capacity, interval, config)
                        matches = result["semantic_sha256"] == repeat["semantic_sha256"]
                        repeat_checks.append({**workload, "configuration": config[0], "matches": matches})
                        if not matches:
                            failure = {"workload": workload, "first": result, "repeat": repeat}
                            (args.output / "repeat-failure.json").write_text(json.dumps(failure, indent=2))
                            raise AssertionError("Fixed-clock repeat changed reuse/accounting/learning")
                    runs.append(result)
                    summaries.append({**workload, **{key: value for key, value in result.items() if key != "requests"}})
                for candidate in runs[1:]:
                    pairs.append({**workload, **paired_summary(runs[0], candidate)})
                raw_path = args.output / f"seed-{seed}-cap-{capacity}-gap-{interval:g}.json"
                raw_path.write_text(json.dumps({"workload": workload, "runs": runs}, indent=2) + "\n")
                print(json.dumps({**workload, "reused_tokens": {run["configuration"]: run["reused_tokens"] for run in runs}}), flush=True)
    if source_hashes() != manifest["source_sha256"]:
        raise AssertionError("Source files changed during the matrix")
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "paired_vs_frozen.json").write_text(json.dumps(pairs, indent=2) + "\n")
    (args.output / "repeat_checks.json").write_text(json.dumps(repeat_checks, indent=2) + "\n")
    manifest.update(
        status="complete",
        finished_at_utc=utc_now(),
        workload_count=len(args.seeds) * len(args.capacities) * len(args.intervals),
        measured_run_count=len(summaries),
        measured_request_count=sum(row["request_count"] for row in summaries),
        repeat_run_count=len(repeat_checks),
        all_repeat_semantics_match=all(row["matches"] for row in repeat_checks) if repeat_checks else None,
        completed_invariants="Capacity, tree/index, parent links, locks, prefix values, eviction accounting, finite weights and single logical observation",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps({"event": "complete", "utc": manifest["finished_at_utc"], "runs": len(summaries)}), flush=True)


if __name__ == "__main__":
    main()
