"""Validate completed replays and summarize client-observed measurements."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(path):
    config = json.loads((path / "config.json").read_text())
    trace_bytes = (path / "trace.json").read_bytes()
    if hashlib.sha256(trace_bytes).hexdigest() != config["trace_sha256"]:
        raise ValueError(f"Trace digest mismatch: {path}")
    trace = json.loads(trace_bytes)
    rows = [
        json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()
    ]
    if len(rows) != config["count"] or any(not row.get("success") for row in rows):
        raise ValueError(f"Incomplete or failed trial: {path}")
    if [row["id"] for row in rows] != list(range(config["count"])):
        raise ValueError(f"Request IDs are not complete and ordered: {path}")
    if len(trace) != len(rows):
        raise ValueError(f"Trace length mismatch: {path}")
    for row, item in zip(rows, trace):
        if row["group"] != item["group"]:
            raise ValueError(f"Trace group mismatch: {path}")
        for field in ("ttft_s", "latency_s"):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError(f"Invalid {field} in {path}")
        if row["meta_info"]["completion_tokens"] != config["output_tokens"]:
            raise ValueError(f"Unexpected completion length in {path}")
        if row["meta_info"].get("finish_reason", {}).get("type") != "length":
            raise ValueError(f"Unexpected finish reason in {path}")
        if row["latency_s"] < row["ttft_s"]:
            raise ValueError(f"TTFT exceeds total latency: {path}")
    # A final server snapshot is required, not just a partial client log.
    before = json.loads((path / "server-before.json").read_text())
    after = json.loads((path / "server-after.json").read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["group"])].append(row)

    def metrics(records):
        queue_times = [
            r["meta_info"]["queue_time"]
            for r in records
            if "queue_time" in r["meta_info"]
        ]
        if any(not math.isfinite(value) or value < 0 for value in queue_times):
            raise ValueError(f"Invalid scheduler queue time: {path}")
        return {
            "requests": len(records),
            "ttft_p50_s": percentile([r["ttft_s"] for r in records], 0.5),
            "ttft_p95_s": percentile([r["ttft_s"] for r in records], 0.95),
            "ttft_p99_s": percentile([r["ttft_s"] for r in records], 0.99),
            "ttft_max_s": max(r["ttft_s"] for r in records),
            "latency_p95_s": percentile([r["latency_s"] for r in records], 0.95),
            "latency_max_s": max(r["latency_s"] for r in records),
            "queue_time_p95_s": percentile(queue_times, 0.95) if queue_times else None,
            "queue_time_max_s": max(queue_times) if queue_times else None,
            "queue_time_field_present": len(queue_times) == len(records),
            "cached_tokens": sum(
                r["meta_info"].get("cached_tokens", 0) for r in records
            ),
            "cached_tokens_field_present": all(
                "cached_tokens" in r["meta_info"] for r in records
            ),
        }

    return {
        "path": str(path),
        "trace_sha256": config["trace_sha256"],
        "concurrency": config.get("concurrency", 1),
        "measurement": "client-observed bounded-wave replay, not serving capacity or starvation proof",
        "all": metrics(rows),
        "groups": {k: metrics(v) for k, v in groups.items()},
        "server_before": before,
        "server_after": after,
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    references = {}
    for path in args.trials:
        summary, rows = summarize(path)
        digest = (summary["trace_sha256"], summary["concurrency"])
        outputs = [r["output_text"] for r in rows]
        if digest in references:
            summary["output_text_mismatches_vs_first_same_trace"] = sum(
                a != b for a, b in zip(references[digest], outputs)
            )
        else:
            references[digest] = outputs
            summary["output_text_mismatches_vs_first_same_trace"] = 0
        results.append(summary)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
