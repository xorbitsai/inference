.. _user_guide_pd_separation:

PD separation
=============

Prefill/Decode (PD) separation is available in the community edition. Prefill
replicas compute prompt KV cache; decode replicas reuse it through Xavier or native NIXL and
produce the response. Requests are scheduled round-robin within each role.
No enterprise package or License is required.

Launch
------

vLLM Xavier P/D uses GPU-to-GPU handoff by default. It requires Linux, NVIDIA GPUs, vLLM >= 0.21.0 and ``xoscar[nixl]>=0.11.1`` in both worker and model environments. Use a reachable host address (not ``0.0.0.0``). Missing NIXL fails the launch; there is no CPU fallback.

The example starts one prefill replica on GPU 0 and one decode replica on GPU 1.
Replace the worker address with the full ``ip:port`` reported by
``client.get_workers_info()``. Both replicas must use the same model and engine
configuration.

.. code-block:: python

   from xinference.client import Client

   client = Client("http://192.168.1.10:9997")
   client.launch_model(
       model_uid="qwen-pd",
       model_name="qwen2.5-instruct",
       model_size_in_billions="0_5",
       model_format="pytorch",
       quantization="none",
       model_engine="vLLM",
       replica=2,
       replica_config=[
           {
               "role": "prefill",
               "devices": [{"worker_ip": "192.168.1.10:9997",
                            "n_gpu": 1, "gpu_idx": [0]}],
           },
           {
               "role": "decode",
               "devices": [{"worker_ip": "192.168.1.10:9997",
                            "n_gpu": 1, "gpu_idx": [1]}],
           },
       ],
   )

The same ``replica_config`` is accepted by ``POST /v1/models``, the async Python
client, and the CLI ``--replica_config`` JSON option. In the Web UI, select vLLM, SGLang or MLX,
enable per-replica placement, and choose Prefill or Decode for each replica.
The community launch path defaults to Xavier when P/D roles appear.

SGLang with Xavier
------------------

SGLang >= ``0.5.21`` accepts the same prefill/decode ``replica_config`` through the Python clients, REST API, CLI and Web UI. Change ``model_engine`` in the example to ``"SGLang"``. Explicit roles automatically enable Xavier; multiple prefill and decode replicas are supported.

SGLang Xavier P/D transfers KV directly between GPUs through Xavier's NIXL transport. Prefill and decode run concurrently using SGLang's native P/D lifecycle. Source and destination GPU slots remain owned until transfers complete; decode also receives the first token sampled by prefill. Streaming and non-streaming responses are supported. Transfer failures raise errors.

The model limitations are described in :ref:`user_guide_vllm_enhancement`. Install ``xoscar[nixl]>=0.11.1`` in worker and model environments and use reachable worker addresses. Missing NIXL fails launch without CPU fallback. Use the ``xavier`` transport; SGLang's own Mooncake backend is not exposed by this integration. For cross-engine P/D, use the per-replica engine settings below.

SGLang GPU P/D does not use ``xavier_cache_bytes`` or CPU HiCache. Retained Xavier GPU history is not yet supported; ``xavier_gpu_cache_bytes`` may be omitted or set to ``0``. Relaunch the deployment after a worker restart. Measure TTFT and throughput against ordinary replicas and native SGLang P/D for your workload.

Set ``XINFERENCE_SGLANG_XAVIER_TRANSFER_TIMEOUT`` in the worker environment or launch ``envs`` to change the Xavier handoff wait limit (default: 600 seconds). Increase it for long prefill queues or large prompts. Unfinished directory rooms expire after this interval; completed directory records expire after this interval unless the active decode request renews its lease. Both roles release them on completion.

SGLang Xavier P/D requires ``disaggregation_decode_enable_radix_cache=false``; each request transfers its full prompt KV. Prefill's local radix cache remains available. Enabling decode radix caching is rejected at launch.

For a native SGLang comparison, keep the same deployment and set ``transfer_backend_type="nixl"``. This uses SGLang's NIXL backend through the same Xinference API and P/D router. Install ``nixl`` in each model environment. The initial native integration requires TP=PP=DP=1, one worker per replica, text requests, and no LoRA, speculative decoding or HiCache. Bootstrap ports are allocated per replica; both replica hosts must be reachable.

Native SGLang P/D requires one tokenizer worker; its internal HTTP endpoints use automatic per-replica authentication.

For fixed-length benchmarks, completion and chat requests accept ``ignore_eos=true``. Override model stop strings with an unused ``stop`` marker and verify the actual output token counts.

MLX with Xavier
---------------

MLX Xavier requires Apple silicon and ``mlx-lm>=0.31.2``. It supports unquantized, full-attention Qwen2, Qwen3 and Llama text models with one worker per replica. Weights and KV cache use FP16. Quantized weights or KV cache, rotating caches, hybrid attention, multimodal inputs, LoRA and speculative decoding are unsupported.

Use ``model_engine="MLX"``, ``model_format="mlx"`` and ``quantization="none"`` with the same prefill/decode roles. The example places one replica on each of two Mac workers. Replace the registered worker addresses with those reported by ``client.get_workers_info()``.

.. code-block:: python

   client.launch_model(
       model_uid="qwen-mlx-pd",
       model_name="qwen2.5-instruct",
       model_size_in_billions="0_5",
       model_engine="MLX",
       model_format="mlx",
       quantization="none",
       replica=2,
       xavier_cache_bytes=536870912,
       replica_config=[
           {"role": "prefill", "devices": [{"worker_ip": "MAC_P_WORKER:PORT",
                                          "n_gpu": "auto"}]},
           {"role": "decode", "devices": [{"worker_ip": "MAC_D_WORKER:PORT",
                                         "n_gpu": "auto"}]},
       ],
   )

MLX Xavier stores immutable CPU pages in a shared actor on the supervisor and transfers them through actor RPC. ``xavier_cache_bytes`` bounds this deployment cache (default: 512 MiB); model weights and per-replica Metal KV allocations consume additional memory. The prefill replica evaluates all but the last prompt token. Decode imports the prefix, evaluates that last token and samples the first response token. Streaming and non-streaming requests are supported. Active handoffs reserve cache capacity; insufficient capacity, incompatible metadata or transfer failure raises an error. The five-minute handoff deadline starts when prefill reserves capacity, before computing the prefix, and includes prefill and transfer time.

Each 64-token FP16 page uses ``2 * 64 * num_layers * num_kv_heads * head_dim * 2`` bytes (K and V, with two bytes per value). For example, 36 layers, 8 KV heads and a head dimension of 128 require 9 MiB per page; 512 MiB holds 56 pages, or 3584 prefix tokens. Round each prefix up to a whole page and size ``xavier_cache_bytes`` for the distinct pages reserved by all concurrent P/D requests. Ordinary shared-cache requests skip publication when their prefix exceeds the cache capacity or the 4096-page protocol limit.

Use the default ``xavier`` transport for MLX; native NIXL is unsupported. All workers must reach the supervisor's actor address. Relaunch after a worker restart. Two replicas on one Mac share its Metal GPU; measure your workload before expecting a throughput benefit. Cross-engine P/D involving MLX, including NVIDIA-to-Mac handoff, is not yet supported.

Cross-engine vLLM/SGLang P/D
--------------------------------------------------------------------------------

Set ``model_engine`` and ``engine_config`` on each ``replica_config`` entry to select vLLM or SGLang independently for each role. Cross-engine P/D requires Xavier GPU transport. Engine options inherit launch arguments; ``engine_config`` overrides them per replica.

.. code-block:: python

   client.launch_model(
       model_uid="qwen-cross-pd",
       model_name="qwen2.5-instruct",
       model_size_in_billions="0_5",
       model_format="pytorch",
       quantization="none",
       model_engine="vLLM",
       replica=2,
       replica_config=[
           {
               "role": "prefill", "model_engine": "vLLM",
               "engine_config": {"max_model_len": 8192},
               "devices": [{"worker_ip": "WORKER_ADDRESS",
                            "n_gpu": 1, "gpu_idx": [0]}],
           },
           {
               "role": "decode", "model_engine": "SGLang",
               "engine_config": {"context_length": 8192},
               "devices": [{"worker_ip": "WORKER_ADDRESS",
                            "n_gpu": 1, "gpu_idx": [1]}],
           },
       ],
   )

Both workers require identical local weights, tokenizer assets and context limits. Supported models are unquantized FP16 full-attention Qwen2, Qwen3 and Llama text models with TP=PP=DP=1. Requires vLLM >=0.21.0, SGLang >=0.5.21 and ``xoscar[nixl]>=0.11.1``. Xavier checks contracts and prompt token IDs, converts 64-token KV pages on GPU and raises errors on incompatibility or transfer failure.

Streaming and non-streaming requests require ``n=1``. SGLang decode uses the first output token sampled by prefill. vLLM decode imports all but the final prompt token, computes that token and samples the output. Engine kernels can produce small numerical differences. vLLM decode requires at least two prompt tokens.

Cross-engine P/D supports no CPU fallback, retained Xavier history, logprobs, structured sampling, LoRA, speculative decoding, multimodal inputs or hybrid attention. Omit ``xavier_gpu_cache_bytes`` or set it to ``0``; do not set ``xavier_cache_bytes``. Relaunch after a worker restart. Native NIXL comparisons use the same engine on both roles. MLX is unsupported.

Native NIXL backend
-------------------

Set ``vllm_transfer_backend_type="nixl"`` in the launch example to use vLLM's
native NixlConnector. Install ``nixl`` in the model environment together with
vLLM 0.21.0 or newer. The initial integration supports text-only models without
LoRA and requires TP=1, PP=1 and DP=1 per replica. Multiple P and D replicas are
supported. Xavier remains the default backend.

Native NIXL PD requires ``n=1`` per request; parallel sampling is not supported.

For NIXL, wildcard worker binds are supported through automatic host discovery. Set ``VLLM_NIXL_SIDE_CHANNEL_HOST`` in launch ``envs`` or the worker environment to select a reachable interface; launch ``envs`` take precedence. If discovery fails or returns loopback, configure this variable explicitly. Xinference always replaces ``VLLM_NIXL_SIDE_CHANNEL_PORT`` with a dynamically allocated port per replica, including on recovery. A fixed side-channel port is not supported; firewalls must allow dynamic ports between replica hosts.

CLI launch
----------

Start a local server first, or use an existing supervisor endpoint:

.. code-block:: bash

   xinference-local --host 127.0.0.1 --port 9997

In another terminal, launch a 1P+1D deployment:

.. code-block:: bash

   xinference launch --endpoint http://127.0.0.1:9997 \
     --model-uid qwen-pd --model-name qwen3 \
     --model-engine vLLM --size-in-billions 0_6 \
     --model-format pytorch --quantization none \
     --replica-config '[
       {"role":"prefill","devices":[{"worker_ip":"WORKER_ADDRESS","n_gpu":1,"gpu_idx":[0]}]},
       {"role":"decode","devices":[{"worker_ip":"WORKER_ADDRESS","n_gpu":1,"gpu_idx":[1]}]}
     ]'

Replace ``WORKER_ADDRESS`` with the worker's registered address, including its
actor port (shown in the worker startup log), not the HTTP API port. For workers
on different machines, use each worker's own address and GPU indices.
``--replica_config`` is an alias for ``--replica-config``. The replica count is
inferred from the array length; no separate Xavier switch is needed. Do not
combine per-replica placement with global ``--n-gpu``, ``--gpu-idx``, or
``--worker-ip``. Invalid PD roles or incomplete P/D topology fail before the
launch request is sent.

Configuration rules
-------------------

* ``replica`` is the total number of P and D replicas. At least one of each role
  is required; more than one of either role is supported.
* Omitted roles default to ``hybrid`` (ordinary replicas). A PD deployment cannot
  mix hybrid replicas with P/D replicas.
* Each entry specifies one worker and its GPU allocation. Different replicas
  may run on different workers. Per-replica placement cannot be combined with
  global ``worker_ip``, ``gpu_idx`` or ``n_gpu``, or cross-worker sharding within
  a replica (``n_worker > 1``).
* ``vllm_transfer_backend_type`` (alias ``transfer_backend_type``) selects
  ``xavier`` (default) or ``nixl``. Unknown backend names are rejected.
* The topology is fixed for the deployment lifetime. To change its size or roles,
  terminate the model and relaunch it with the new configuration.

Inference and lifecycle
-----------------------

Send OpenAI-compatible chat/completion requests to the model UID used at launch. Both streaming and non-streaming responses are supported. Wait for all replicas to be ready before sending requests. Aborting a request stops work on both roles; terminating the model releases the deployment resources.

After a worker restart, relaunch the native NIXL PD deployment.

Hybrid/recurrent attention limitation
-------------------------------------

Qwen3.5 text requests are supported with Xavier and native NIXL. Set ``language_model_only=True``, ``enable_prefix_caching=False``, ``mamba_cache_mode="none"``, ``disable_hybrid_kv_cache_manager=False`` and ``async_scheduling=False`` on both replicas. Use TP=1 and PP=1. Xavier history caching is not available for these hybrid/recurrent attention models; their configured history budget is not allocated. Image/video inputs and speculative decoding are not supported in this configuration. Native NIXL requires vLLM >= 0.22.0 for this mode.

vLLM Xavier requirements and memory
-----------------------------------

Xavier V1 requires one GPU per replica (TP=1, PP=1) and text-only models without LoRA. Multimodal models, prompt embeddings and salted prompts are not supported.

Xavier's CPU cache can consume as much memory as the engine's GPU KV cache. Allow additional host memory for transfers when sizing your deployment.

Direct handoff and tiered history
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``xavier_gpu_cache_bytes`` defaults to ``268435456`` (256 MiB per replica) for P/D. History uses GPU first and spills to CPU when GPU capacity is exhausted. CPU history is bounded by the engine KV block capacity and can approach that cache size in host memory. CPU hits restore KV to GPU; misses compute locally. ``0`` disables history, retaining direct transfer. Transfer failures raise errors.

Direct handoff currently requires ``n=1``.

History is restored on P before the handoff to D. Long prompts may retain only a prefix; cache retention is best effort.

Unclaimed handoffs expire after 120 seconds. Claimed handoffs expire after 10 minutes without a scheduling retry or transfer progress. Active copies finish before release. Expired handoffs are recomputed locally on D.

The budget applies only to Xavier's GPU cache. Leave additional GPU memory for the model, vLLM's own KV cache and transfers. When this cache fills, Xavier stores reusable data in CPU memory; if no cache space is available, new data may not be retained for reuse.

Direct transfers and history loads can overlap with response generation for other requests.

Transfer failures and incompatible KV cache layouts raise errors instead of silently recomputing the request.

For implementation details, profiling and contributor tests, see the `Xavier developer README <https://github.com/xorbitsai/inference/blob/main/xinference/model/llm/vllm/xavier/README.md>`_.
