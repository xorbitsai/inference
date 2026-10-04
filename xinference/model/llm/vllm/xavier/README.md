# Xavier developer notes

This document covers implementation, diagnostics and contributor validation. For installation, replica placement, supported models and cache settings, see the [PD separation user guide](../../../../../doc/source/user_guide/pd_separation.rst).

## Backend boundaries

Xavier and vLLM's native `NixlConnector` are separate backends. Selecting `vllm_transfer_backend_type="nixl"` uses the native connector; configuring `xavier_gpu_cache_bytes` does not select it.

The Xavier V1 connector supports vLLM 0.21.0 or newer. P/D requires xoscar[nixl]>=0.11.1 in both worker and model environments. The legacy V0 adapter remains for non-P/D deployments with vLLM below 0.11.0; versions 0.11–0.20 are unsupported. GPU integration CI pins vLLM 0.21.0 and NIXL 1.1.0. Match vLLM, Torch and NIXL versions: importing vLLM alone does not validate engine startup.

The native backend passes producer handoff metadata to the selected decoder and lets NIXL manage transfer completion and cache release. Failed handoffs fail the request rather than silently recomputing. `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` bounds producer retention if a decoder never consumes a completed prefill. This path does not create Xavier collective actors or force eager execution. Parallel sampling is rejected because child requests cannot safely share one producer transfer lease.

## Cache ownership and transfer

P/D uses request-scoped GPU-to-GPU handoff by default, with no opt-in flag. The producer keeps its engine blocks until decoder reads and optional history copies finish. Independent history uses 256 MiB of GPU storage per replica by default and spills to CPU, bounded by engine KV block capacity. `xavier_gpu_cache_bytes=0` disables history while keeping direct handoff. CPU-history hits restore to the local GPU in per-layer batches; P-to-D traffic remains xoscar NIXL. Missing NIXL fails launch.

Unclaimed producer tickets expire after 120 seconds. D atomically claims a live ticket before reporting a remote hit; expiry is a scheduler miss, not a worker transfer error. Claimed tickets have a 600-second idle lease refreshed by scheduling retries and slab transfers. Active reads cannot expire. If a claim expires before or between slabs, the worker reports every destination block of that request through vLLM's load-error callback together with receive completion; the default Xavier recompute policy retries locally without exposing partial KV or crashing EngineCore. Other transfer/layout failures still propagate. Allocation marks the handoff consumed, so preempted decode requests recompute locally rather than rereading the ticket. Router cleanup abandons only unclaimed tickets, while allocated loads drain before release.

History uses one writer, a 64 MiB candidate-prefix cap (also limited by total history capacity), and a 10 ms soft deadline. Long prompts retain the leading admissible prefix instead of being skipped entirely; all engine blocks remain owned until the active chunk drains. `history_capacity_limited_blocks` counts candidates beyond the cap, `history_deadline_dropped_blocks` and `history_closing_dropped_blocks` count remaining candidates abandoned at the deadline or shutdown, and `history_skipped_requests` counts a busy writer. When both tiers are full, new content must be observed again before replacing retained blocks. Admission stops at the first rejected or capacity-limited prefix block. Cache and probation LRU order retain prefix heads longer than tails. History hit counters count unique blocks actually written by successful loads, not reservations or scheduling retries. Active leases prevent eviction. History is restored on P; routing still visits P before D. Direct handoff supports `n=1`.

The following snapshot-path details apply to hybrid deployments. Xavier CPU snapshots are keyed by prompt content, not reusable engine block IDs. Readers reserve complete snapshots during transfer; unavailable snapshots or cache pressure can cause a cache miss and local computation. This differs from a transfer failure, which propagates as an error. Only complete layers are published for remote reuse. CPU capacity is bounded by the engine's KV block capacity.

In the GPU-first path, engine caches are shared with the local TransferActor through CUDA IPC. Retained GPU snapshots use xoscar NIXL; CPU snapshots use batched Gloo transfers. The worker sets `UCX_MEMTYPE_CACHE=n` and defaults `UCX_TLS` to `tcp,cuda_copy,cuda_ipc`, preserving an explicit `UCX_TLS` setting.

The snapshot budget excludes the model, engine KV cache, persistent transfer buffers, temporary tensors and allocator overhead. Separate send and receive slabs allow a replica to serve a peer while loading from another. Slabs normally use up to 16 MiB each; a single larger block can require more. Persistent small views share their allocations. Each source rank must produce consecutive small batches before selecting the small view, avoiding repeated NIXL re-registration for alternating full/small batches.

GPU overflow demotes unleased snapshots to CPU. If both tiers have no evictable space, new snapshots are skipped. Layout mismatches and transfer failures propagate instead of silently converting data or promising recomputation. If staging fails, outstanding CUDA work is fenced before unpublished snapshots are discarded; synchronization errors still propagate.

P/D always uses asynchronous loads. In the hybrid snapshot path, when `xavier_gpu_cache_bytes` is set (including zero), KV loads execute asynchronously in the TransferActor so ready requests can continue decoding. Omitting the setting for hybrid replicas retains synchronous CPU loading. Cancellation holds destination blocks and source leases until writes complete. Cleanup attempts every lease release and preserves the original load exception if cleanup also fails. Shutdown drains outstanding operations before releasing buffers and snapshot storage, and logs cache placement and transfer counters. Repeated close calls share one close task; the closed runtime remains a guard against remapping.

If the last request is aborted while an async load is pending, an idle vLLM EngineCore may not poll completion again until the next request arrives. Actor-side writes and source lease release continue independently, but destination block reclamation and connector-side completion cleanup wait for that next engine step or shutdown.

The connector accepts supported block-first and K/V-first layouts and rejects an ambiguous block axis. Shared null-block positions in allocated cache groups are omitted from destination mappings; conflicting writes to real destinations remain errors. Qwen3.5 text-only P/D additionally transfers each Gated DeltaNet cache group independently. Launch with `language_model_only=True`, `enable_prefix_caching=False`, `mamba_cache_mode="none"`, and `async_scheduling=False`. For recurrent models, Xavier defaults prefix caching and async scheduling off and rejects enabling either. On vLLM 0.21 it enables the hybrid KV cache manager for recurrent models only; explicit user settings are preserved. Keep that manager enabled for recurrent handoff. TP/PP remain 1 and speculative decoding is rejected.

For recurrent models, P computes the first N-1 prompt tokens; D loads that exact recurrent state and computes the last token. P truncation is idempotent across scheduling retries. Attention tensors may contain several physical kernel blocks per logical block; zero-copy views keep those physical blocks within a single logical transfer unit. Tickets authorize source blocks separately for each layer, including both convolution and SSM state tensors. The engine retains all cache groups until transfer completion. One-token prompts recompute on D without transferring state.

Recurrent history reuse is currently disabled, even when a history budget is configured. Ordinary attention-block hashes cannot identify a reusable recurrent state at an arbitrary prefix boundary. Supporting historical reuse requires atomic snapshots of attention KV plus the recurrent state at the same token position. Full-attention history caching remains unchanged.

In the legacy V0 path, prefill replicas release only the requested completed sequence; ordinary hybrid and decode replicas retain automatic cleanup.

## Request routing and lifecycle

The PD router issues a one-output-token prefill subrequest without changing the decoder's generation settings. Routing becomes available only after the replicas and transfer components are ready. Aborts reach both roles. Deployment termination also removes the PD router and Xavier's rank-zero coordinator, collective manager and block tracker.

Native NIXL routes are owned by the supervisor, so worker restart requires deployment relaunch; model-subprocess recovery on a live worker re-registers its PD route. Each native replica receives a dynamically allocated side-channel port, including after recovery. Host discovery and explicit interface configuration are documented in the user guide.

## Profiling

Set `XINFERENCE_XAVIER_PROFILE=1` in the model process environment and collect server logs. From the repository root:

```bash
python benchmark/analyze_pd_profile.py server.log --output profile.json
```

Profiling synchronizes CUDA and adds logging overhead. Run it separately from throughput benchmarks. Timings are nested: `load_rpc` includes actor timings, and `actor_receive` includes `gloo_receive`. Do not add nested totals or interpret Gloo receive wait as isolated network time.

## Backend comparison

From the repository root:

```bash
python benchmark/benchmark_pd.py --help
```

Supply launch JSON with model settings and P/D placement, plus a JSONL workload of chat request bodies. The runner compares ordinary hybrid replicas, Xavier and native NIXL sequentially using the same GPU allocation and workload. It records per-request results, TTFT, average time per output token after the first token (TPOT), latency percentiles and throughput under the selected latency limits.

Repeated workload entries exercise warm prefix reuse; distinct prefixes exercise cold requests. Verify actual cache-hit counters before attributing gains to Xavier history rather than the engine's own cache. Two GPUs cover 1P1D versus two hybrid replicas; 2P2D requires four independent replica GPUs. Record model/runtime versions, cache budgets and hardware with results. Diagnostic timings and historical measurements should not be presented as current-head serving benchmarks.

## Two-GPU integration test

### Multiple P/D replicas sharing GPUs

Run the opt-in multi-replica test with a small full-attention model:

```bash
XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU=1 XINFERENCE_TEST_PD_MULTI_GPU=1 \
  python -m pytest -v xinference/model/llm/vllm/xavier/test/test_pd_multi_gpu.py
```

The test covers 2P1D, 1P2D and 2P2D for Xavier and native NIXL. All producers
share GPU 0 and all decoders share GPU 1, with a 0.35 engine memory budget per
replica. The existing `XINFERENCE_TEST_PD_MODEL_PATH`, model name and size
overrides apply. It checks request-specific answers, streaming, eight concurrent
requests, actual KV loads and Xavier history restoration with engine prefix
caching disabled. For 2P2D it also removes and re-registers one decoder route:
equal-length round robins otherwise cover only two of the four P/D pairs.
This changes route registration, not the decoder process, and is not a crash
recovery test. `MULTI_PD_RESULT` records observed pairs and request counts.

This setup validates multi-replica behavior on two GPUs. The colocated engines
share compute and memory bandwidth, so its performance does not establish
four-GPU scaling.

### One producer and one decoder

Use two free NVIDIA GPUs and the pinned vLLM environment, with NIXL installed. Run from the repository root:

```bash
XINFERENCE_TEST_PD_GPU=1 python -m pytest -v \
  xinference/model/llm/vllm/xavier/test/test_pd_gpu.py
```

The test runs default Xavier direct handoff with tiered history, Xavier with history disabled, and native NIXL in real subprocesses, with one GPU per role. It checks streaming and non-streaming responses, repeated prompts and four concurrent requests, then terminates the deployment. Local prefix caching is disabled. Exact repeated-text checks are restricted to native NIXL and Xavier with history disabled; fixed-prefix and raw-bit history tests cover history reuse separately. Xavier requires producer registration and successful async completion logs from `get_finished`, and the default-history case requires a positive P-side history restore for each repeated prompt; native NIXL requires `calling _read_blocks` and completed receive logs, including the concurrent requests.

For locally cached weights, set `XINFERENCE_TEST_PD_MODEL_PATH`, `XINFERENCE_TEST_PD_MODEL_NAME` and `XINFERENCE_TEST_PD_MODEL_SIZE` to the path and registered model name/size. The manually triggered **PD GPU integration** GitHub Actions workflow runs the same test on a selected two-GPU runner.

### Qwen3.5 recurrent-state regression

Set `XINFERENCE_TEST_PD_RECURRENT_GPU=1` and
`XINFERENCE_TEST_PD_RECURRENT_MODEL_PATH` to a local Qwen3.5-0.8B checkpoint,
then run `test/test_pd_recurrent_gpu.py`. This opt-in test compares Xavier and
native NIXL P/D against standalone greedy outputs for short and long prompts,
including repeated and concurrent requests. It requires actual transfer logs.
The test uses BF16, TP=PP=1, text-only mode and no prefix caching; it does not
establish recurrent history-cache support or a throughput advantage.

Native NIXL's GDN support landed in vLLM PR #41869 and is included in
vLLM 0.22.0. Stock vLLM 0.21.0 rejects GDN in its convolution-state transfer
setup. The native leg of the regression requires 0.22.0 or newer; Xavier's
custom handoff continues to support 0.21.0.
