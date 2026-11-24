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

GPU serving smoke validation passed on 2026-11-23; the initial four-policy GPU
comparison completed on 2026-11-24 (see results below). The local isolated runtime
uses PyTorch 2.9.1+cu130, Transformers
4.57.1, FlashInfer 0.5.3 and the official sgl-kernel 0.3.19 CUDA 13 wheel.
The inherited base image's newer FlashInfer JIT cache caused the first launch to
fail; disabling system-site-package inheritance fixed that dependency conflict.
CUDA matrix multiplication and sgl-kernel import passed on the RTX 4070 Laptop.
Qwen3-0.6B loaded BF16 weights (reported 1.22 GB) and allocated exactly 2048 KV
tokens (reported 0.11 GB each for K and V). Two identical 162-token inputs each
completed 16 output tokens; reported cached tokens were 0 and 161 respectively.
An additional 24-request streaming warmup completed. These are smoke checks, not
evidence that any research policy improves latency, throughput or fairness.

The Windows launcher
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

The launcher also accepts `-Image`, `-PythonExecutable`, and `-ModelPath`. For the
local research image, use `sglang-prefix-research:cu130-v056post2-isolated` and
`/opt/sglang-research/bin/python`. An explicit local model snapshot path avoids
optional Hugging Face metadata queries during offline startup. The local runtime
image is not an upstream SGLang release image; record its image ID and dependency
freeze with every experiment.

```sh
python3 benchmark/prefix_value_replay.py --output /tmp/trial-lru --seed 1729
python3 benchmark/summarize_prefix_value.py /tmp/trial-lru --output /tmp/summary.json
```

For a running server, the unified entrypoint checks its policy and actual token
capacity, waits up to 300 seconds for `/server_info`, replays 24 warmup requests,
flushes the cache, checks reset research counters, and records a 240-request run:

```sh
python3 benchmark/run_prefix_value_trial.py --policy lru --capacity 2048 --seed 1729 --output /tmp/trial-lru
```

It requires fresh trial and warmup output directories and records the reset
response and counters in `protocol.json`. A failed warmup or replay stops the run.

`--concurrency 4` runs fixed waves of up to four requests, waiting for each wave
to finish before submitting the next. Warmup uses the same concurrency. Every
request records actual client dispatch/completion offsets; within-wave network
arrival order can vary. All outcomes in a failed wave are saved before stopping.
This tests bounded concurrent cache pressure, not sustained overload or maximum
capacity. Summaries include maximum TTFT, end-to-end P95/maximum, and scheduler
queue-time P95/maximum for every prefix group. Missing queue metadata is marked,
not silently treated as zero. Output comparisons are grouped by both trace and
concurrency because different batching can change numerical execution.

For this local setup, `run_prefix_value_followups.ps1 -OutputRoot <new-directory>`
reuses the four existing research containers at the recorded image ID. It runs a
second sequential pass in reverse policy order and then a four-policy wave-4
pass. It pins seed 1729/capacity 2048, requires a clean committed checkout, saves
the plan and container logs, stops each server in a finally block, and validates
the completed replays. These follow-up results are not included in the initial
single-run table below until the matrix actually completes.

The client requires a normal length finish, expected output-token count and SSE
completion marker. Summary generation validates the trace digest and request
coverage. Same-trace output text mismatches are reported, not silently discarded.
The workload is sequential; its measured rate is not maximum serving capacity.
GPU prefill time, concurrency/SLO behavior and economic cost require further
measurement beyond this initial protocol.

Novelty review against RadixAttention, learned prefix caching and newer retention
policies is still required. CPU reuse improvements alone do not pass that gate.

## Initial GPU comparison (2026-11-24)

The [machine-readable result](results/prefix_value_gpu_20261124.json) records the
tested commit, image ID, model snapshot, shared settings, trace digest, per-group
metrics and runtime counters. Each policy completed the same 240-request trace
after 24 warmup requests and a verified counter reset, with capacity 2048 tokens.
The observed execution order was learned, LRU, indexed LRU, then frozen.

| Policy | Successful requests | Reused tokens | TTFT P50 (ms) | TTFT P95 (ms) | Observation time (ms total) | Eviction time (ms total) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LRU | 240/240 | 37,120 | 120.15 | 167.01 | 0 | 37.94 |
| Indexed LRU | 240/240 | 37,120 | 110.39 | 157.88 | 0 | 34.46 |
| Frozen value | 240/240 | 37,120 | 110.71 | 155.90 | 61.61 | 99.37 |
| Learned value | 240/240 | 37,120 | 117.76 | 171.63 | 64.72 | 106.85 |

All policies matched LRU's cached-token count for every request, not only the
aggregate, and produced identical output text. Each recorded 122 eviction calls
and 73,136 evicted tokens. Learned and frozen policies each observed 240 logical
requests, produced 1,212 positive and 328 negative prefix-boundary labels, and
retained 480 history entries. Learned weights changed to approximately
`[1.679, 0.895, -2.691, 0.731]`; frozen weights remained zero.

This confirms end-to-end online updates, but **does not establish a performance
benefit**. Learning did not improve observed reuse and added bookkeeping cost;
its P95/P99 TTFT were higher than LRU in this run. Indexed LRU's eviction-path
saving was only about 3.5 ms across the entire trial, so the much larger client
latency difference cannot be attributed to that saving alone. These are single
runs with an unrandomized order, one seed, one capacity and sequential arrivals.
Repeated paired trials, other capacities/arrival patterns and concurrent fairness
tests remain necessary for broader performance claims. Do not report these
numbers as serving capacity, measured GPU prefill time, production savings, or
proof of starvation freedom.

All 37 RadixCache unit tests (including the added indexed/value tests)
also passed directly under the installed runtime, using its real imports rather
than the earlier CPU-only HostKVCache import shim. Test allocator tensors remain
CPU tensors: this strengthens dependency/lifecycle validation, not GPU allocator
coverage. The earlier 2026-11-23 LRU run is retained separately and was not used
as the same-day latency comparison.

## Repeated and concurrent GPU follow-ups (2026-11-24)

The [follow-up audit](results/prefix_value_gpu_followups_20261124.json) covers a
second sequential pass in reverse policy order and one wave-4 pass, all at seed
1729 and capacity 2048. These eight trials completed 1,920/1,920 measured requests.
The source was pinned to `94bf350`; model, runtime image and effective settings
were checked. Client tests were extended to seven cases, including actual thread
overlap and preserving peer outcomes when one stream fails.

| Policy | Sequential repeat TTFT P95 (ms) | Sequential reused tokens | Wave-4 TTFT P95 (ms) | Wave-4 reused tokens | Wave-4 text differences vs LRU |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 158.72 | 37,120 | 869.32 | 34,208 | reference |
| Indexed LRU | 145.81 | 37,120 | 903.36 | 32,624 | 23/240 |
| Frozen value | 147.09 | 37,120 | 886.51 | 32,144 | 19/240 |
| Learned value | 151.71 | 37,120 | 896.64 | 38,624 | 27/240 |

Sequential output text again matched across policies. Learned P95 was worse than
LRU in the first pass and better in the second; reuse remained identical. This
does not establish a stable latency benefit. Indexed LRU's total eviction time
was 35.68 ms versus LRU's 39.82 ms in the repeat, a small absolute difference.

Concurrent runs are **not controlled arrival-order comparisons**. Among 60 waves,
server arrival order differed from LRU in 47 indexed, 39 frozen, and 50 learned
waves. Do not attribute the learned run's extra reused tokens solely to learning.
Its P95 TTFT was higher than LRU despite the extra reuse. All runs had zero
reported retractions, all prefix groups completed, and maximum reported scheduler
queue times were respectively 0.884, 0.837, 0.830 and 0.780 seconds for LRU,
indexed, frozen and learned. Group P95s sometimes regressed substantially; the
audit retains them. Some groups have only four requests, so their percentile
estimates are weak. Finite completion is not a starvation bound or an SLO result.

A separate LRU-against-itself wave-4 repeat also completed 240/240 requests, but
produced 21 text differences and 50/60 changed arrival orders. Reuse changed from
34,208 to 32,112 tokens, and P95 TTFT from 869.32 to 971.31 ms. Its maximum client
TTFT was 5.393 seconds, while maximum scheduler queue time was 0.896 seconds;
the client stall is retained without claiming a cause. This demonstrates baseline
variability, but does **not** by itself prove that every policy-related text
difference is harmless. Concurrent output equivalence remains unverified.

`analyze_prefix_value_campaign.py <trial-directories...> --output <new-json>`
validates full response coverage, matching settings and reset counters, and uses
the first LRU run for each trace/concurrency/capacity/backend/determinism group.
It records output-difference IDs, per-group timings, retractions and server
arrival-order differences. It never compares deterministic and ordinary runs as
one configuration.

The launcher accepts `-DeterministicInference` for a separate correctness
diagnostic using the pinned version's Triton batch-invariant path and PyTorch
sampler. Two initial diagnostics failed before measurement with a 512-token
prefill budget. The first timed out at 180 seconds; a 600-second warmup retry did
not resolve the problem. Compilation workers were present, but a consistent
Python stack showed the scheduler repeatedly checking admission, not waiting in
compilation. Source inspection established the mismatch:
`Scheduler.init_deterministic_inference_config()` defaults Triton truncation
alignment to 4096, while `PrefillAdder.add_one_req()` rejects a truncated chunk
smaller than that alignment. With a 512 budget, the trace's longer prompts can
remain unadmitted even on LRU. Both failed warmups and the diagnostic stack are
retained. No measured result was accepted from either attempt.

The launcher now uses a 4096-token prefill budget **only** for deterministic
diagnostics, keeping the ordinary experiment's 512 budget unchanged. Corrected
LRU/learned results are pending. The separate warmup-timeout option remains useful
for cold compilation, but was not the fix for this admission problem. Do not mix
these deterministic timings with the ordinary configuration in either table.

## Preliminary related-work gate (2026-11-23)

The following primary publication pages and abstracts were checked. This is an
overlap screen, not a full paper/code reproduction or an exhaustive review.

| Work | Reported mechanism | Overlap and implication for this prototype |
| --- | --- | --- |
| [LPC, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/414f642a1ea9350006669774cba9bcd4-Abstract-Conference.html) | Predict conversational continuation from content and combine predictions with access timestamps. | Learned prefix eviction combined with recency is already published. Using history features instead of content does not by itself establish novelty. |
| [SAECache, arXiv 2605.18825](https://arxiv.org/abs/2605.18825) | Task-specific queues, semantic token-type weights, and online adaptation from eviction feedback. | Online adaptive reuse-value estimation already has close prior work; a lightweight learner alone is insufficient as a contribution. |
| [When Fancy Eviction Fails, arXiv 2609.28870v3](https://arxiv.org/abs/2609.28870v3) | Production-trace study; recency baseline with one-hit demotion, compute-aware eviction and capacity-adaptive granularity. | Recency plus compute cost is also covered. Preserve LRU controls, one-off traffic, different capacities and negative results. |

Current decision: **the broad novelty claim does not pass**. Continue the local
prototype as an engineering experiment, without calling it a new research
algorithm. A narrower contribution would need evidence of a specific unresolved
structural dependency or maintenance-cost problem, full-text and code comparison,
and controlled experiments against relevant implementations. The indexed leaf
set and split-stable identities are implementation choices, not established
research novelty. Published performance claims have not been reproduced here.
