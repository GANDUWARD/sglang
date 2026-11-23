# Experimental prefix retention on SGLang

This branch studies prefix retention under limited KV capacity on a single RTX
4070 Laptop GPU using Qwen3-0.6B. The current implementation is a prototype, not a
validated serving optimization or a claimed novel cache algorithm.

## Fixed baseline and implementation

The source baseline is SGLang `v0.5.6.post2`, commit
`5c8bd8b51b53b9b39eb1edec582ee43b21002106`. The matching candidate runtime image is
`lmsysorg/sglang:v0.5.6.post2`. This historical release was selected for an initial
fixed experiment, not as a substitute for comparison with newer SGLang policies.

The implementation retains SGLang's existing radix prefix sharing and physical
KV storage. It does not claim to introduce prefix deduplication.

| Policy | Candidate structure | Retention score |
| --- | --- | --- |
| `lru` | Original tree traversal | Original last-access timestamp |
| `indexed_lru` | Incrementally maintained structural leaf set | Original timestamp |
| `reuse_value_frozen` | Leaf set | Bounded value bonus with frozen predictor |
| `reuse_value` | Leaf set | Bounded value bonus with online predictor |

The leaf set includes locked leaves and filters them when collecting eviction
candidates. Insertion removes a parent from the set; deleting the last child adds
its parent, except the root. Splitting an edge preserves the existing descendant
leaf. Eviction still builds a heap over candidates and may proceed toward the
root. The change removes full-tree traversal; it does not eliminate candidate
scoring or heap construction.

Research modes support plain Python RadixCache without EAGLE. Hybrid caches,
HiCache, C++ cache implementations and disaggregated serving are not validated.
The scheduler observation hook currently covers ordinary, non-disaggregated
request admission only.

## Predictor and value definition

Each accepted logical request is observed once, independently of KV cache hits.
A marker on the request prevents rescheduling from counting as another access.
Namespace-aware BLAKE2b prefix identities are sampled every 64 tokens, so their
identity survives radix edge splitting and physical KV eviction.

The bounded history holds at most 4096 identities. An observation is labeled
positive if that prefix recurs within the next 64 logical request observations,
including the boundary. Otherwise it becomes negative after the window expires.
History entries removed before their outcomes are known are censored, not treated
as negative. A run ending with pending observations does not manufacture labels.

A four-parameter logistic predictor uses a bias, decayed frequency, previous
reuse interval and prefix depth. It updates by SGD on cross entropy with a small
weight penalty. It runs in Python on CPU, without scheduling a model-training GPU
kernel. Current learning rate is 0.03; feature normalization and all constants are
experimental rather than tuned results.

For each candidate leaf, the prototype uses its deepest sampled prefix identity.
The value proxy is predicted reuse probability times request-age decay times
`1 + depth / 4096`. The latter is a dimensionless per-token recomputation proxy,
not measured prefill cost. It assumes equal KV bytes per token within this model;
it does not sum all descendant benefits or model arbitrary heterogeneous blocks.

The eviction score is `last_access_time + min(value, 1 second)`. Thus a score
cannot indefinitely pin an unlocked leaf against sufficiently newer candidates.
This is not a proof of request fairness or a guarantee against scheduler starvation.
Low-volume prefix groups and phase changes must be evaluated separately.

## Validation completed before GPU experiments

- 37 real RadixCache CPU semantic tests pass, including index consistency after
  splits, locking, eviction and reset, and bounded learning-policy scores.
- 6 dependency-free predictor tests pass, including delayed labels, censoring,
  namespace isolation, interval features and decay after a hotspot stops.
- 5 HTTP client protocol tests pass using synthetic streams. These are not model
  inference tests.
- An 18-case CPU replay matrix uses seeds 1729/1730/1731, capacities 2048/4096
  tokens and logical arrival gaps 0.01/0.1/1.0 seconds, with 240 requests per case.
  Relative reused-token changes for the learned policy versus LRU are 10 positive,
  6 equal and 2 negative cases, ranging from -2.04% to +11.25%.
  Indexed LRU matches LRU's reused-token count in all 18 cases.

CPU semantic tests isolate the unused HostKVCache import because the temporary
test container has mismatched CUDA libraries. Instantiating that substitute raises
an error. The real RadixCache, real Torch tensors and cache operations are used;
CUDA kernels, physical GPU allocation and serving performance are not validated.

In this baseline's plain RadixCache, `hit_count` is initialized but never incremented.
Its LFU strategy therefore degenerates to LRU. The replay records this limitation;
LFU is not an independent strong baseline here.

An earlier replay used elapsed wall time and was sensitive to interpreter speed.
Its negative result is retained in the local evidence directory, but the fixed
logical-clock matrix is the reproducible semantic comparison. Neither is a GPU
latency or throughput measurement.

## Reproduce CPU validation

From the repository root, in an environment with SGLang's Python dependencies:

```sh
export PYTHONPATH="$PWD/python"
python3 test/srt/test_reuse_value.py
python3 test/srt/test_prefix_value_client.py
python3 test/srt/run_radix_cpu_tests.py
python3 benchmark/prefix_value_cpu_matrix.py --output /tmp/prefix-value-matrix
```

Use a new output directory: scripts refuse to overwrite an existing experiment.
All matrix per-request results, traces and summary rows are retained. CPU operation
timings include replay bookkeeping and must not be presented as scheduler timings.

## GPU experiment procedure

GPU serving and performance comparison remain pending. The Windows launcher
`benchmark/run_prefix_value_server.ps1` uses Docker GPU passthrough, the pinned
image, a read-only source mount and the existing `vllm-hf-cache` model volume.
It requests 2048-token contexts, four running requests, 512-token prefill chunks,
Triton attention, disabled CUDA Graphs and static memory fraction 0.65.

1. Inspect the image ID and loaded source path, then start one policy at a time.
2. Confirm readiness, actual token capacity and model configuration from logs and
   `/server_info`; first verify an ordinary model request and repeated-prefix hits.
3. Warm up consistently, then reset the cache before measuring each fresh trial.
4. Replay identical token traces against LRU, indexed LRU, frozen and learned
   policies, preserving negative results and per-group regressions.
5. Check complete responses, output comparisons, effective settings and research
   counters before comparing client TTFT and latency.

```powershell
./benchmark/run_prefix_value_server.ps1 -Policy lru -CacheTokens 2048
```

```sh
python3 benchmark/prefix_value_replay.py --output /tmp/trial-lru --seed 1729
python3 benchmark/summarize_prefix_value.py /tmp/trial-lru --output /tmp/summary.json
```

The client requires a normal length finish, expected output-token count and SSE
completion marker. Summary generation validates the trace digest and request
coverage. Same-trace output text mismatches are reported, not silently discarded.
The workload is sequential; its measured rate is not maximum serving capacity.
GPU prefill time, concurrency/SLO behavior and economic cost require further
measurement beyond this initial protocol.

Novelty review against RadixAttention, learned prefix caching and newer retention
policies is still required. CPU reuse improvements alone do not pass that gate.
