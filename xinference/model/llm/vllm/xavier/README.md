# Xavier developer notes

This document covers implementation, diagnostics and contributor validation. For installation, replica placement, supported models and cache settings, see the [PD separation user guide](../../../../../doc/source/user_guide/pd_separation.rst).

## Backend boundaries

Xavier and vLLM's native `NixlConnector` are separate backends. Selecting `vllm_transfer_backend_type="nixl"` uses the native connector; configuring `xavier_gpu_cache_bytes` does not select it.

The Xavier V1 connector supports vLLM 0.21.0 or newer. The legacy V0 adapter remains for vLLM below 0.11.0; versions 0.11–0.20 are unsupported. GPU integration CI pins vLLM 0.21.0 and NIXL 1.1.0. Match vLLM, Torch and NIXL versions: importing vLLM alone does not validate engine startup.

The native backend passes producer handoff metadata to the selected decoder and lets NIXL manage transfer completion and cache release. Failed handoffs fail the request rather than silently recomputing. `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` bounds producer retention if a decoder never consumes a completed prefill. This path does not create Xavier collective actors or force eager execution. Parallel sampling is rejected because child requests cannot safely share one producer transfer lease.

## Cache ownership and transfer

Xavier CPU snapshots are keyed by prompt content, not reusable engine block IDs. Readers reserve complete snapshots during transfer; unavailable snapshots or cache pressure can cause a cache miss and local computation. This differs from a transfer failure, which propagates as an error. Only complete layers are published for remote reuse. CPU capacity is bounded by the engine's KV block capacity.

In the GPU-first path, engine caches are shared with the local TransferActor through CUDA IPC. Retained GPU snapshots use xoscar NIXL; CPU snapshots use batched Gloo transfers. The worker sets `UCX_MEMTYPE_CACHE=n` and defaults `UCX_TLS` to `tcp,cuda_copy,cuda_ipc`, preserving an explicit `UCX_TLS` setting.

The snapshot budget excludes the model, engine KV cache, persistent transfer buffers, temporary tensors and allocator overhead. Separate send and receive slabs allow a replica to serve a peer while loading from another. Slabs normally use up to 16 MiB each; a single larger block can require more. Persistent small views share their allocations. Each source rank must produce consecutive small batches before selecting the small view, avoiding repeated NIXL re-registration for alternating full/small batches.

GPU overflow demotes unleased snapshots to CPU. If both tiers have no evictable space, new snapshots are skipped. Layout mismatches and transfer failures propagate instead of silently converting data or promising recomputation. If staging fails, outstanding CUDA work is fenced before unpublished snapshots are discarded; synchronization errors still propagate.

KV loads execute asynchronously in the TransferActor so ready requests can continue decoding. Cancellation holds destination blocks and source leases until writes complete. Cleanup attempts every lease release and preserves the original load exception if cleanup also fails. Shutdown drains outstanding operations before releasing buffers and snapshot storage, and logs cache placement and transfer counters. Repeated close calls share one close task; the closed runtime remains a guard against remapping.

The connector accepts supported block-first and K/V-first layouts and rejects an ambiguous block axis. Shared null-block positions in allocated cache groups are omitted from destination mappings; conflicting writes to real destinations remain errors. This is not a claim of support for recurrent state: hybrid/recurrent attention models such as Qwen3.5 are rejected. Successful model launch alone does not demonstrate correct recurrent-state transfer.

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

Use two free NVIDIA GPUs and the pinned vLLM environment, with NIXL installed. Run from the repository root:

```bash
XINFERENCE_TEST_PD_GPU=1 python -m pytest -v \
  xinference/model/llm/vllm/xavier/test/test_pd_gpu.py
```

The test runs both Xavier and native NIXL in real subprocesses, with one GPU per role. It checks streaming and non-streaming responses, repeated prompts and four concurrent requests, then terminates the deployment. Xavier requires producer staging and decoder KV-load log evidence; native NIXL requires `calling _read_blocks` and completed receive logs, including the concurrent requests.

For locally cached weights, set `XINFERENCE_TEST_PD_MODEL_PATH`, `XINFERENCE_TEST_PD_MODEL_NAME` and `XINFERENCE_TEST_PD_MODEL_SIZE` to the path and registered model name/size. The manually triggered **PD GPU integration** GitHub Actions workflow runs the same test on a selected two-GPU runner.
