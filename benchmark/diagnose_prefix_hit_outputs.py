"""Replay identical targets with verified cold/hot prefixes on a dedicated local server.

Use --output <new-directory>. POST /flush_cache resets the local research server
before *each* condition. This is an output diagnostic, not a latency benchmark.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from prefix_value_replay import replay_one
from summarize_prefix_value import summarize


FIXED = (
    "version", "model_path", "tokenizer_path", "revision", "dtype", "kv_cache_dtype",
    "page_size", "context_length", "attention_backend", "disable_cuda_graph",
    "disable_radix_cache", "tp_size", "pp_size", "dp_size", "schedule_policy",
    "max_running_requests", "random_seed",
)
EXECUTION = ("enable_deterministic_inference", "sampling_backend", "chunked_prefill_size")


def server_info(url):
    with urllib.request.urlopen(url + "/server_info", timeout=10) as response:
        return json.load(response)


def signature(info):
    states = info["internal_states"]
    if len(states) != 1 or info["disable_radix_cache"]:
        raise ValueError("Expected a single worker with radix caching enabled")
    stats = states[0]["prefix_value_research"]
    return {
        **{key: info[key] for key in FIXED + EXECUTION},
        "capacity": states[0]["memory_usage"]["token_capacity"],
        "policy": stats["policy"],
        "rate": stats.get("learning_rate"),
        "formula": stats.get("value_formula"),
    }


def flush(url, settings):
    # HTTP 400 means the scheduler still has running or waiting requests.
    deadline = time.monotonic() + 20
    while True:
        try:
            request = urllib.request.Request(url + "/flush_cache", method="POST")
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"Flush returned HTTP {response.status}")
                message = response.read().decode()
            state = server_info(url)
            if signature(state) != settings:
                raise RuntimeError("Server settings changed during diagnostic")
            stats = state["internal_states"][0]["prefix_value_research"]
            if (stats["observations"] == stats["eviction_calls"] == 0
                    and stats.get("history_entries", 0) == 0
                    and not any(stats.get("predictor_weights", []))):
                return {"response": message, "reset_stats": stats}
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
        if time.monotonic() >= deadline:
            raise RuntimeError("Dedicated server did not confirm an idle, reset radix cache")
        time.sleep(0.2)


def make_primer(item, depth):
    tokens = list(item["input_ids"])
    if not 0 <= depth < len(tokens):
        raise ValueError("Priming depth must be shorter than the target")
    # Reuse a token ID known to be valid for this exact checkpoint.
    tokens[depth] = next(token for token in tokens if token != tokens[depth])
    return {**item, "input_ids": tokens}


def load_targets(baseline, candidate):
    base, old_rows = summarize(baseline)
    other, new_rows = summarize(candidate)
    if base["trace_sha256"] != other["trace_sha256"] or base["concurrency"] != 1 or other["concurrency"] != 1:
        raise ValueError("The source trials must use the same sequential trace")
    configs = [json.loads((path / "config.json").read_bytes()) for path in (baseline, candidate)]
    if configs[0]["output_tokens"] != configs[1]["output_tokens"]:
        raise ValueError("Original generation lengths differ")
    trace = json.loads((baseline / "trace.json").read_bytes())
    targets = []
    for item, original, changed in zip(trace, old_rows, new_rows):
        originals = {"baseline": original, "candidate": changed}
        hits = [r["meta_info"]["cached_tokens"] for r in originals.values()]
        if min(hits) == 0 and max(hits) > 0:
            targets.append((item, max(hits), originals))
    if not targets:
        raise ValueError("No original target has both a zero-hit and nonzero-hit observation")
    return [signature(s["server_before"]) for s in (base, other)], base["trace_sha256"], configs[0]["output_tokens"], targets


def run(args):
    url = args.url.rstrip("/")
    if urlsplit(url).hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Cache flush diagnostics require a dedicated local endpoint")
    original_settings, trace_hash, output_tokens, targets = load_targets(args.baseline_trial, args.candidate_trial)
    settings = signature(server_info(url))
    for original in original_settings:
        for key in FIXED + ("capacity",):
            if settings[key] != original[key]:
                raise ValueError(f"Diagnostic {key} differs from original trial")
        if original["enable_deterministic_inference"] == settings["enable_deterministic_inference"]:
            if any(settings[key] != original[key] for key in EXECUTION):
                raise ValueError("Sampling or prefill settings changed within the same execution mode")
    args.output.mkdir(parents=True, exist_ok=False)
    orders = (("cold", "hot"), ("hot", "cold"))
    inputs = {
        "scope": "Dedicated service, paired greedy text comparison with verified prefix lengths",
        "url": url, "trace_sha256": trace_hash, "server": settings,
        "original_trials": [str(args.baseline_trial), str(args.candidate_trial)],
        "original_servers": original_settings, "condition_orders": orders,
        "target_output_tokens": output_tokens, "primer_output_tokens": 1,
        "targets": [
            {"request": item, "hit_depth": depth,
             "originals": {arm: {"output_text": row["output_text"], "meta_info": row["meta_info"]}
                           for arm, row in originals.items()}}
            for item, depth, originals in targets
        ],
    }
    (args.output / "protocol.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    summary = []
    with (args.output / "observations.jsonl").open("x", encoding="utf-8") as log:
        def record(event):
            log.write(json.dumps(event) + "\n")
            log.flush()

        for item, hot_depth, originals in targets:
            repeats = {"cold": [], "hot": []}
            for cycle, order in enumerate(orders, 1):
                for condition in order:
                    depth = 0 if condition == "cold" else hot_depth
                    case = {"target_id": item["id"], "cycle": cycle, "condition": condition}
                    record({**case, "stage": "flush", **flush(url, settings)})
                    primer = make_primer(item, depth)
                    for stage, request, count, expected_hits in (
                        ("primer", primer, 1, 0), ("target", item, output_tokens, depth),
                    ):
                        row = replay_one(url, request, count)
                        hits = row["meta_info"]["cached_tokens"]
                        record({**case, "stage": stage, "expected_hits": expected_hits, **row})
                        if hits != expected_hits or row["meta_info"]["prompt_tokens"] != len(request["input_ids"]):
                            raise RuntimeError(f"Unexpected cache hit or prompt length at {case} {stage}")
                    state = server_info(url)
                    stats = state["internal_states"][0]["prefix_value_research"]
                    record({**case, "stage": "verified_pair", "stats": stats})
                    if signature(state) != settings or stats["eviction_calls"] != 0:
                        raise RuntimeError(f"Configuration changed or eviction occurred at {case}")
                    if "history_entries" in stats and stats["observations"] != 2:
                        raise RuntimeError(f"Unexpected logical request count at {case}")
                    repeats[condition].append(row["output_text"])
            summary.append({
                "target_id": item["id"], "hot_depth": hot_depth, "outputs": repeats,
                "repeat_agreement": {k: vals[0] == vals[1] for k, vals in repeats.items()},
                "cold_hot_text_equal_by_cycle": [a == b for a, b in zip(repeats["cold"], repeats["hot"])],
                "matches_original_by_condition": {
                    k: {arm: [txt == row["output_text"] for txt in vals] for arm, row in originals.items()}
                    for k, vals in repeats.items()
                },
            })
    if signature(server_info(url)) != settings:
        raise RuntimeError("Diagnostic server changed configuration")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(row["target_id"], "cold vs hot text equal in two orders:", row["cold_hot_text_equal_by_cycle"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--baseline-trial", type=Path, required=True)
    parser.add_argument("--candidate-trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
