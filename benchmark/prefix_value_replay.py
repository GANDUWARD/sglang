"""Sequential synthetic token replay for paired prefix-cache experiments.

Use a fresh server for each policy/repeat. Identical seed produces identical
input IDs. This measures low-concurrency cache competition, not serving capacity.
"""

import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path


def make_trace(seed, count):
    rng = random.Random(seed)
    groups = {
        i: [rng.randrange(1000, 15000) for _ in range(256 * (1 + i % 3))]
        for i in range(12)
    }
    trace = []
    for i in range(count):
        # A phase change replaces the hot set; every fifth request is pollution.
        if i % 5 == 0:
            group = "one_off"
            tokens = [rng.randrange(1000, 15000) for _ in range(512)]
        else:
            phase = 0 if i < count // 2 else 6
            group = phase + rng.choices(range(6), weights=[8, 4, 2, 1, 1, 1])[0]
            tokens = list(groups[group])
        tokens += [rng.randrange(16000, 20000) for _ in range(16)]
        trace.append({"id": i, "group": group, "input_ids": tokens})
    return trace


def replay_one(url, item, output_tokens):
    body = json.dumps(
        {
            "input_ids": item["input_ids"],
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_tokens,
                "ignore_eos": True,
            },
        }
    ).encode()
    request = urllib.request.Request(
        url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    first_token = None
    last = None
    done = False
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            meta = event.get("meta_info", {})
            if meta.get("completion_tokens", 0) > 0 and first_token is None:
                first_token = time.perf_counter() - start
            last = event
    if last is None or first_token is None:
        raise RuntimeError("No token-bearing stream event received")
    meta = last.get("meta_info", {})
    if not done or not isinstance(meta.get("finish_reason"), dict):
        raise RuntimeError("Stream ended without completion marker and finish reason")
    if meta["finish_reason"].get("type") != "length":
        raise RuntimeError(f"Unexpected finish reason: {meta['finish_reason']}")
    if meta.get("completion_tokens") != output_tokens:
        raise RuntimeError(f"Unexpected completion length: {meta}")
    return {
        "id": item["id"],
        "group": item["group"],
        "ttft_s": first_token,
        "latency_s": time.perf_counter() - start,
        "meta_info": meta,
        "output_text": last.get("text", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trace = make_trace(args.seed, args.count)
    encoded = json.dumps(trace, separators=(",", ":")).encode()
    (args.output / "trace.json").write_bytes(encoded)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["trace_sha256"] = hashlib.sha256(encoded).hexdigest()
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    if args.trace_only:
        return

    def save_server_info(name):
        with urllib.request.urlopen(args.url + "/server_info", timeout=30) as response:
            info = json.load(response)
        (args.output / name).write_text(json.dumps(info, indent=2))

    save_server_info("server-before.json")
    with (args.output / "requests.jsonl").open("w", encoding="utf-8") as log:
        for item in trace:
            try:
                result = replay_one(args.url, item, args.output_tokens)
                result["success"] = True
            except Exception as error:
                result = {"id": item["id"], "success": False, "error": repr(error)}
                log.write(json.dumps(result) + "\n")
                log.flush()
                raise
            log.write(json.dumps(result) + "\n")
            log.flush()
    save_server_info("server-after.json")


if __name__ == "__main__":
    main()
