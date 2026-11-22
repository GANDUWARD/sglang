"""Replay real RadixCache on CPU; no GPU timing or serving claims.

Host KV transfers are explicitly unavailable, as in run_radix_cpu_tests.py.
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import Mock

host = types.ModuleType("sglang.srt.mem_cache.memory_pool_host")


class UnavailableHostKVCache:
    def __init__(self, *args, **kwargs):
        raise AssertionError("CPU replay cannot use host GPU transfers")


host.HostKVCache = UnavailableHostKVCache
sys.modules[host.__name__] = host

import torch

from sglang.srt.mem_cache import radix_cache as radix_module
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--interval-s", type=float, default=0.1)
    args = parser.parse_args(argv)
    trace = json.loads(args.trace.read_text())
    if max(len(r["input_ids"]) for r in trace) > args.capacity:
        raise ValueError("Each request must fit in the cache")
    results = {}
    clock = [0.0]
    # Only the cache module receives logical time; operation timings stay real.
    radix_module.time = types.SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter
    )
    for policy in ("lru", "lfu", "indexed_lru", "reuse_value_frozen", "reuse_value"):
        clock[0] = 0.0
        allocator = Mock()
        allocator.device = torch.device("cpu")
        evicted = [0]
        allocator.free.side_effect = lambda values: evicted.__setitem__(
            0, evicted[0] + len(values)
        )
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                eviction_policy=policy,
            )
        )
        rows = []
        for item in trace:
            clock[0] += args.interval_s
            tokens = item["input_ids"]
            start = time.perf_counter()
            cache.observe_request(
                types.SimpleNamespace(origin_input_ids=tokens, extra_key=None)
            )
            match = cache.match_prefix(RadixKey(tokens))
            reused = len(match.device_indices)
            cache.inc_lock_ref(match.last_device_node)
            required = max(0, cache.total_size() + len(tokens) - reused - args.capacity)
            if required:
                cache.evict(required)
            cache.insert(RadixKey(tokens), torch.tensor(tokens))
            cache.dec_lock_ref(match.last_device_node)
            if cache.total_size() > args.capacity:
                raise AssertionError("Capacity invariant violated")
            rows.append(
                {
                    "id": item["id"],
                    "group": item["group"],
                    "reused": reused,
                    "input_tokens": len(tokens),
                    "cpu_operation_s": time.perf_counter() - start,
                }
            )
        model = cache.reuse_model
        results[policy] = {
            "reused_tokens": sum(r["reused"] for r in rows),
            "evicted_tokens": evicted[0],
            "positive_labels": model.positive_labels if model else None,
            "negative_labels": model.negative_labels if model else None,
            "weights": model.weights if model else None,
            "requests": rows,
            "cache_stats": cache.get_research_stats(),
        }
    args.output.write_text(
        json.dumps(
            {
                "validation": "CPU semantic replay only",
                "interval_s": args.interval_s,
                "lfu_note": "upstream plain RadixCache never increments hit_count; LFU degenerates to LRU here",
                "capacity_tokens": args.capacity,
                "results": results,
            },
            indent=2,
        )
    )
    for policy, result in results.items():
        print(policy, result["reused_tokens"], result["evicted_tokens"])


if __name__ == "__main__":
    main()
