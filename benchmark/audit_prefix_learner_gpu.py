"""Validate a paired learner-only campaign and emit portable summary rows."""

import argparse
import json
from pathlib import Path

from summarize_prefix_value import percentile, summarize


def audit(root):
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8-sig"))
    study = plan.get("Study", "learner")
    expected_order = {
        "learner": ["default", "frozen", "fast", "fast", "frozen", "default"],
        "value": ["full", "no_depth", "no_depth", "full"],
    }[study]
    assert [entry["Label"] for entry in plan["Order"]] == expected_order
    assert plan["Concurrency"] == 1 and plan["Seed"] == 1735
    common_fields = (
        "model_path", "dtype", "kv_cache_dtype", "context_length", "page_size",
        "chunked_prefill_size", "mem_fraction_static", "schedule_policy",
        "disable_radix_cache", "max_running_requests", "attention_backend",
        "sampling_backend", "enable_deterministic_inference", "disable_cuda_graph",
        "random_seed", "tp_size", "pp_size",
    )
    reference = None
    settings = None
    rows = []
    for index, case in enumerate(plan["Order"], start=1):
        path = root / f"{index:02d}-{case['Label']}"
        summary, requests = summarize(path)
        config = json.loads((path / "config.json").read_text())
        protocol = json.loads((path / "protocol.json").read_text())
        assert config["seed"] == plan["Seed"] and config["output_tokens"] == 16
        assert summary["concurrency"] == 1 and len(requests) == 240
        assert protocol["warmup_count"] == 24 and protocol["capacity"] == 2048
        for when in ("server_before", "server_after"):
            state = summary[when]
            observed = {field: state[field] for field in common_fields}
            if settings is None:
                settings = observed
            assert settings == observed
            assert state["model_path"] == plan["Model"] and not state["disable_radix_cache"]
            internal = state["internal_states"][0]
            assert internal["memory_usage"]["token_capacity"] == plan["Capacity"]
            stats = internal["prefix_value_research"]
            assert stats["policy"] == state["radix_eviction_policy"] == case["Policy"]
            assert stats["learning_rate"] == state["prefix_value_learning_rate"] == case["Rate"]
            if study == "value":
                assert stats["value_formula"] == state["prefix_value_formula"] == case["Formula"] == case["Label"]
                assert case["Rate"] == 0.03 and case["Policy"] == "reuse_value"
            if when == "server_before":
                assert stats["observations"] == stats["eviction_calls"] == 0
                assert stats["predictor_weights"] == [0] * 4
            else:
                assert stats["observations"] == 240 and stats["eviction_calls"] > 0
                assert (stats["predictor_weights"] == [0] * 4) == (case["Label"] == "frozen")
        assert summary["all"]["cached_tokens_field_present"]
        if reference is None:
            reference = requests
            reference_trace = summary["trace_sha256"]
            reference_learning = {field: stats[field] for field in ("predictor_weights", "positive_labels", "negative_labels", "censored_labels")}
        if study == "value":
            assert all(stats[field] == value for field, value in reference_learning.items())
        assert summary["trace_sha256"] == reference_trace
        arrivals = [r["meta_info"]["request_received_ts"] for r in requests]
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        assert all(gap > 0 for gap in gaps)
        rows.append({
            "trial": path.name, "policy": case["Policy"], "rate": case["Rate"],
            "metrics": summary["all"], "groups": summary["groups"], "stats": stats,
            "interarrival_s": {key: percentile(gaps, q) for key, q in
                               (("p10", 0.10), ("p50", 0.50), ("p90", 0.90))},
            "text_difference_ids_vs_first": [
                r["id"] for r, b in zip(requests, reference) if r["output_text"] != b["output_text"]
            ],
            "hit_difference_ids_vs_first": [
                r["id"] for r, b in zip(requests, reference)
                if r["meta_info"]["cached_tokens"] != b["meta_info"]["cached_tokens"]
            ],
        })
    return {"plan": plan, "trace_sha256": reference_trace, "settings": settings, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
