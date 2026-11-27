"""Warm up, reset, and measure one already-running local research server."""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--expected-learning-rate", type=float)
    parser.add_argument("--expected-value-formula")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--warmup-request-timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    warmup = args.output.with_name(args.output.name + "-warmup")
    if args.output.exists() or warmup.exists():
        raise FileExistsError("Trial and warmup directories must both be new")

    def info():
        with urllib.request.urlopen(args.url + "/server_info", timeout=5) as response:
            return json.load(response)

    deadline = time.monotonic() + 300
    while True:
        try:
            state = info()
            break
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    assert state["radix_eviction_policy"] == args.policy, "Wrong server policy"
    states = state["internal_states"]
    assert len(states) == 1, "Only single-worker experiments are supported"
    assert states[0]["memory_usage"]["token_capacity"] == args.capacity
    replay = Path(__file__).with_name("prefix_value_replay.py")
    common = [
        sys.executable,
        str(replay),
        "--url",
        args.url,
        "--concurrency",
        str(args.concurrency),
    ]
    subprocess.run(
        common
        + [
            "--count",
            "24",
            "--seed",
            "1729",
            "--output",
            str(warmup),
            "--request-timeout",
            str(args.warmup_request_timeout),
        ],
        check=True,
    )
    request = urllib.request.Request(args.url + "/flush_cache", method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        flush_result = response.read().decode()
    after_reset = info()
    stats = after_reset["internal_states"][0]["prefix_value_research"]
    if args.expected_learning_rate is not None:
        assert stats["learning_rate"] == args.expected_learning_rate, "Wrong learner rate"
        assert stats["predictor_weights"] == [0.0] * 4, "Learner was not reset"
    if args.expected_value_formula is not None:
        assert stats["value_formula"] == args.expected_value_formula, "Wrong value formula"
    assert (
        stats["observations"] == 0 and stats["eviction_calls"] == 0
    ), "Cache statistics did not reset"
    print("Warmup and reset passed; starting 240-request trial", flush=True)
    subprocess.run(
        common
        + ["--count", "240", "--seed", str(args.seed), "--output", str(args.output)],
        check=True,
    )
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "policy": args.policy,
                "capacity": args.capacity,
                "warmup_count": 24,
                "warmup_request_timeout_s": args.warmup_request_timeout,
                "warmup_seed": 1729,
                "concurrency": args.concurrency,
                "warmup_directory": str(warmup),
                "flush_response": flush_result,
                "reset_stats": stats,
            },
            indent=2,
        )
    )
    print("Trial complete", args.output, flush=True)


if __name__ == "__main__":
    main()
