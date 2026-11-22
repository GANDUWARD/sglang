"""CPU semantic tests only; explicitly exclude unused Host KV GPU transfers.

This loads the real RadixCache and tests, but it does NOT validate CUDA kernels,
GPU allocation or serving. Any attempt to instantiate HostKVCache fails loudly.
Use the matching SGLang container for unmodified imports and GPU validation.
"""

import runpy
import sys
import types
from pathlib import Path


class UnavailableHostKVCache:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Host KV operations are excluded from CPU semantic tests")


if __name__ == "__main__":
    host = types.ModuleType("sglang.srt.mem_cache.memory_pool_host")
    host.HostKVCache = UnavailableHostKVCache
    sys.modules[host.__name__] = host
    print("CPU SEMANTIC TEST: HostKVCache isolated; no GPU validation", flush=True)
    runpy.run_path(
        str(Path(__file__).with_name("test_radix_cache_unit.py")), run_name="__main__"
    )
