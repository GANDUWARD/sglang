"""Fixed, non-tuned semantic matrix; preserve every per-request result."""

import argparse
import json
from pathlib import Path

from prefix_value_cpu_replay import main as replay
from prefix_value_replay import make_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for seed in (1729, 1730, 1731):
        trace_path = args.output / f"trace-{seed}.json"
        trace_path.write_text(json.dumps(make_trace(seed, 240)))
        for capacity in (2048, 4096):
            for interval in (0.01, 0.1, 1.0):
                output = args.output / f"seed-{seed}-cap-{capacity}-gap-{interval}.json"
                replay(
                    [
                        "--trace",
                        str(trace_path),
                        "--output",
                        str(output),
                        "--capacity",
                        str(capacity),
                        "--interval-s",
                        str(interval),
                    ]
                )
                result = json.loads(output.read_text())["results"]
                base = result["lru"]["reused_tokens"]
                learned = result["reuse_value"]["reused_tokens"]
                row = {
                    "seed": seed,
                    "capacity": capacity,
                    "interval_s": interval,
                    "lru_reused": base,
                    "learned_reused": learned,
                    "relative_change": learned / base - 1,
                    "indexed_matches_lru": result["indexed_lru"]["reused_tokens"]
                    == base,
                }
                summary.append(row)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
                print("CASE", row, flush=True)


if __name__ == "__main__":
    main()
