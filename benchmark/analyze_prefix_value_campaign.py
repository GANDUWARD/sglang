"""Validate trials and compare each configuration against its first LRU run."""

import argparse
import json
from pathlib import Path

from summarize_prefix_value import summarize


def configuration(summary):
    server = summary["server_before"]
    capacity = server["internal_states"][0]["memory_usage"]["token_capacity"]
    return (
        summary["trace_sha256"],
        summary["concurrency"],
        capacity,
        server["attention_backend"],
        server["sampling_backend"],
        server["enable_deterministic_inference"],
        server["disable_cuda_graph"],
    )


def wave_orders(rows, field, concurrency):
    orders = []
    for start in range(0, len(rows), concurrency):
        wave = rows[start : start + concurrency]
        if any(field not in row["meta_info"] for row in wave):
            return None
        orders.append(
            [r["id"] for r in sorted(wave, key=lambda r: r["meta_info"][field])]
        )
    return orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [summarize(path) for path in args.trials]
    references = {}
    for summary, rows in loaded:
        if summary["server_before"]["radix_eviction_policy"] == "lru":
            references.setdefault(configuration(summary), (summary, rows))
    results = []
    common_fields = [
        "context_length",
        "max_running_requests",
        "chunked_prefill_size",
        "mem_fraction_static",
        "page_size",
        "kv_cache_dtype",
        "dtype",
        "schedule_policy",
        "random_seed",
        "tp_size",
        "pp_size",
    ]
    for summary, rows in loaded:
        baseline, reference_rows = references[configuration(summary)]
        for field in common_fields:
            assert (
                summary["server_before"][field] == baseline["server_before"][field]
            ), field
            assert (
                summary["server_after"][field] == summary["server_before"][field]
            ), field
        assert configuration(
            dict(summary, server_before=summary["server_after"])
        ) == configuration(summary)
        before = summary["server_before"]["internal_states"][0]["prefix_value_research"]
        after = summary["server_after"]["internal_states"][0]["prefix_value_research"]
        assert before["policy"] == after["policy"]
        assert before["observations"] == 0 and before["eviction_calls"] == 0
        if after["policy"].startswith("reuse_value"):
            assert after["observations"] == len(rows)
        mismatches = [
            r["id"]
            for r, b in zip(rows, reference_rows)
            if r["output_text"] != b["output_text"]
        ]
        orders = wave_orders(rows, "request_received_ts", summary["concurrency"])
        baseline_orders = wave_orders(
            reference_rows, "request_received_ts", summary["concurrency"]
        )
        results.append(
            {
                "trial": Path(summary["path"]).name,
                "baseline": Path(baseline["path"]).name,
                "policy": after["policy"],
                "concurrency": summary["concurrency"],
                "capacity": configuration(summary)[2],
                "deterministic": summary["server_before"][
                    "enable_deterministic_inference"
                ],
                "trace_sha256": summary["trace_sha256"],
                "metrics": summary["all"],
                "per_group": summary["groups"],
                "runtime_counters": after,
                "output_mismatch_ids_vs_baseline": mismatches,
                "cached_token_mismatch_count_vs_baseline": sum(
                    r["meta_info"]["cached_tokens"] != b["meta_info"]["cached_tokens"]
                    for r, b in zip(rows, reference_rows)
                ),
                "server_arrival_order_different_waves": (
                    None
                    if orders is None or baseline_orders is None
                    else sum(a != b for a, b in zip(orders, baseline_orders))
                ),
                "retractions": sum(
                    r["meta_info"].get("total_retractions", 0) for r in rows
                ),
                "retractions_field_present": all(
                    "total_retractions" in r["meta_info"] for r in rows
                ),
            }
        )
    report = {
        "measured_requests": sum(len(rows) for _, rows in loaded),
        "failed_requests": 0,
        "scope": "Finite synthetic waves, not maximum throughput or starvation freedom",
        "trials": results,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    for result in results:
        metrics = result["metrics"]
        print(
            result["trial"],
            "cached=",
            metrics["cached_tokens"],
            "p95_ms=",
            round(metrics["ttft_p95_s"] * 1000, 2),
            "text_mismatches=",
            len(result["output_mismatch_ids_vs_baseline"]),
            "arrival_order_different_waves=",
            result["server_arrival_order_different_waves"],
        )


if __name__ == "__main__":
    main()
