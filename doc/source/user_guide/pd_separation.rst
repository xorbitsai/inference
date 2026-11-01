.. _user_guide_pd_separation:

PD separation
=============

.. versionadded:: v3.4.0

Prefill/Decode (PD) separation is available in the community edition. Prefill
replicas compute prompt KV cache; decode replicas reuse it through Xavier or native NIXL and
produce the response. Requests are scheduled round-robin within each role.
No enterprise package or License is required.

Choose an engine and backend
----------------------------

For same-engine P/D, choose the engine first, then its Native NIXL or Xavier backend below. vLLM, SGLang and TRT-LLM support both; Xavier is the default. MLX supports Xavier only. Cross-engine deployments use Xavier and are described separately.

Native NIXL uses the engine's own P/D connector. Xavier's GPU transport also uses NIXL, with Xavier managing the handoff. The ``xavier_cache_backend`` setting (``native`` or ``lmcache``) selects prefix-cache storage separately from P/D transport. Starting in v4.2.0, LMCache can provide optional historical prefix reuse on P for all four engines; D continues to use Xavier handoff. See :ref:`xavier-pd-lmcache` for configuration and limits.

Launch
------

Python client
~~~~~~~~~~~~~

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

CLI launch
~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~

Send OpenAI-compatible chat/completion requests to the model UID used at launch. Both streaming and non-streaming responses are supported. Wait for all replicas to be ready before sending requests. Aborting a request stops work on both roles; terminating the model releases the deployment resources.

After a worker restart, relaunch the native NIXL PD deployment.

For fixed-length benchmarks, completion and chat requests accept ``ignore_eos=true``. Override model stop strings with an unused ``stop`` marker and verify the actual output token counts.

.. _user_guide_pd_same_engine:

Same-engine P/D
---------------

vLLM
~~~~

.. _native-nixl-backend:

.. _user_guide_pd_vllm_native:

Native NIXL
^^^^^^^^^^^

.. versionadded:: v4.0.0

Set ``vllm_transfer_backend_type="nixl"`` in the launch example to use vLLM's
native NixlConnector. Install ``nixl`` in the model environment together with
vLLM 0.21.0 or newer. The initial integration supports text-only models without
LoRA and requires TP=1, PP=1 and DP=1 per replica. Multiple P and D replicas are
supported. Xavier remains the default backend.

Native NIXL PD requires ``n=1`` per request; parallel sampling is not supported.

For NIXL, wildcard worker binds are supported through automatic host discovery. Set ``VLLM_NIXL_SIDE_CHANNEL_HOST`` in launch ``envs`` or the worker environment to select a reachable interface; launch ``envs`` take precedence. If discovery fails or returns loopback, configure this variable explicitly. Xinference always replaces ``VLLM_NIXL_SIDE_CHANNEL_PORT`` with a dynamically allocated port per replica, including on recovery. A fixed side-channel port is not supported; firewalls must allow dynamic ports between replica hosts.

.. _vllm-xavier-requirements-and-memory:

.. _user_guide_pd_vllm_xavier:

Xavier (default)
^^^^^^^^^^^^^^^^

.. versionchanged:: v4.0.0

   vLLM Xavier P/D uses GPU-to-GPU handoff by default. It requires Linux, NVIDIA GPUs, vLLM >= 0.21.0 and ``xoscar[nixl]>=0.11.1`` in both worker and model environments. Use a reachable host address (not ``0.0.0.0``). Missing NIXL fails the launch; there is no CPU fallback.

Xavier V1 requires one GPU per replica (TP=1, PP=1) and text-only models without LoRA. Multimodal models and prompt embeddings are not supported. See :ref:`xavier-cache-isolation` for request-scoped cache isolation.

Xavier's CPU cache can consume as much memory as the engine's GPU KV cache. Allow additional host memory for transfers when sizing your deployment.

Direct handoff and tiered history
"""""""""""""""""""""""""""""""""

.. versionadded:: v4.0.0

``xavier_gpu_cache_bytes`` defaults to ``268435456`` (256 MiB per replica) for P/D. History uses GPU first and spills to CPU when GPU capacity is exhausted. CPU history is bounded by the engine KV block capacity and can approach that cache size in host memory. CPU hits restore KV to GPU; misses compute locally. ``0`` disables history, retaining direct transfer. Transfer failures raise errors.

Direct handoff currently requires ``n=1``.

History is restored on P before the handoff to D. Long prompts may retain only a prefix; cache retention is best effort.

Unclaimed handoffs expire after 120 seconds. Claimed handoffs expire after 10 minutes without a scheduling retry or transfer progress. Active copies finish before release. Expired handoffs are recomputed locally on D.

The budget applies only to Xavier's GPU cache. Leave additional GPU memory for the model, vLLM's own KV cache and transfers. When this cache fills, Xavier stores reusable data in CPU memory; if no cache space is available, new data may not be retained for reuse.

Direct transfers and history loads can overlap with response generation for other requests.

Transfer failures and incompatible KV cache layouts raise errors instead of silently recomputing the request.

For implementation details, profiling and contributor tests, see the `Xavier developer README <https://github.com/xorbitsai/inference/blob/main/xinference/model/llm/vllm/xavier/README.md>`_.

Hybrid/recurrent attention limitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: v4.0.0

Qwen3.5 text requests are supported with Xavier and native NIXL. Set ``language_model_only=True``, ``enable_prefix_caching=False``, ``mamba_cache_mode="none"``, ``disable_hybrid_kv_cache_manager=False`` and ``async_scheduling=False`` on both replicas. Use TP=1 and PP=1. Xavier history caching is not available for these hybrid/recurrent attention models; their configured history budget is not allocated. Image/video inputs and speculative decoding are not supported in this configuration. Native NIXL requires vLLM >= 0.22.0 for this mode.

SGLang
~~~~~~

.. _user_guide_pd_sglang_native:

Native NIXL
^^^^^^^^^^^

.. versionadded:: v4.0.0

Set ``model_engine="SGLang"`` and ``transfer_backend_type="nixl"`` with the launch example's P/D roles. This uses SGLang's NIXL backend through the same Xinference API and P/D router. Install ``nixl`` in each model environment. The initial native integration requires TP=PP=DP=1, one worker per replica, text requests, and no LoRA, speculative decoding or HiCache. Bootstrap ports are allocated per replica; both replica hosts must be reachable.

Native SGLang P/D requires one tokenizer worker; its internal HTTP endpoints use automatic per-replica authentication.

.. _sglang-with-xavier:

.. _user_guide_pd_sglang_xavier:

Xavier (default)
^^^^^^^^^^^^^^^^

.. versionadded:: v4.0.0

SGLang >= ``0.5.21`` accepts the same prefill/decode ``replica_config`` through the Python clients, REST API, CLI and Web UI. Change ``model_engine`` in the example to ``"SGLang"``. Explicit roles automatically enable Xavier; multiple prefill and decode replicas are supported.

SGLang Xavier P/D transfers KV directly between GPUs through Xavier's NIXL transport. Prefill and decode run concurrently using SGLang's native P/D lifecycle. Source and destination GPU slots remain owned until transfers complete; decode also receives the first token sampled by prefill. Streaming and non-streaming responses are supported. Transfer failures raise errors.

The model limitations are described in :ref:`user_guide_vllm_enhancement`. Install ``xoscar[nixl]>=0.11.1`` in worker and model environments and use reachable worker addresses. Missing NIXL fails launch without CPU fallback. Use the ``xavier`` transport; SGLang's own Mooncake backend is not exposed by this integration. For cross-engine P/D, use the per-replica engine settings below.

By default, SGLang GPU P/D does not use ``xavier_cache_bytes`` or CPU HiCache. Optional P-side LMCache history enables native HiCache on P only; see :ref:`xavier-pd-lmcache`. Retained Xavier GPU history is not yet supported; ``xavier_gpu_cache_bytes`` may be omitted or set to ``0``. Relaunch the deployment after a worker restart. Measure TTFT and throughput against ordinary replicas and native SGLang P/D for your workload.

Set ``XINFERENCE_SGLANG_XAVIER_TRANSFER_TIMEOUT`` in the worker environment or launch ``envs`` to change the Xavier handoff wait limit (default: 600 seconds). Increase it for long prefill queues or large prompts. Unfinished directory rooms expire after this interval; completed directory records expire after this interval unless the active decode request renews its lease. Both roles release them on completion.

SGLang Xavier P/D requires ``disaggregation_decode_enable_radix_cache=false``; each request transfers its full prompt KV. Prefill's local radix cache remains available. Enabling decode radix caching is rejected at launch.

TRT-LLM
~~~~~~~

.. versionadded:: v4.1.0

Use the same ``replica_config`` with ``model_engine="TRT-LLM"``, ``model_name="qwen3"``, ``model_size_in_billions="0_6"``, ``model_format="pytorch"`` and ``quantization="none"`` through the Python clients, REST API or CLI. Both roles require Linux/CUDA, TensorRT-LLM ``1.2.1``, BF16 dense Qwen3 weights and KV, one worker and one GPU per replica (TP=PP=CP=1). Chunked prefill and overlap scheduling must remain disabled. Native NIXL with different engines, quantization, multimodal input, LoRA, speculative decoding and ``cache_salt`` are unsupported.

Speculative decoding is supported on ordinary TRT-LLM replicas; it cannot be combined with P/D in this adapter. See :ref:`speculative_decoding` for launch options.

.. _user_guide_pd_trt_native:

Native NIXL
^^^^^^^^^^^

For native TRT P/D, set ``transfer_backend_type="nixl"`` and omit ``enable_xavier`` and ``xavier_cache_bytes``. TRT runs a context-only request on prefill, transfers its native context metadata and first sampled token, and resumes a generation-only request on decode using TRT's NIXL transceiver. NIXL must be available in both model environments. ``cache_transceiver_config.kv_transfer_timeout_ms`` defaults to 30000 and can be overridden in the launch arguments. The router releases the prefill request on completion, failure or cancellation. Measure both transports against ordinary replicas for your workload.

A ``cache_transceiver_config.kv_transfer_sender_future_timeout_ms`` value of ``None`` is normalized to 1000 ms so the executor can process cancellation even without a decode receiver. This does not change ``kv_transfer_timeout_ms=None``, which disables the overall transfer deadline.

.. _user_guide_pd_trt_xavier:

Xavier (default)
^^^^^^^^^^^^^^^^

The default Xavier path uses NIXL to transfer live 64-token KV pages, including a partially filled final page, directly between GPUs. A separate transfer process maps the TRT KV pool through CUDA IPC. Source slots stay pinned until GPU reads finish; decode waits for writes and replays the final prompt token. Install ``xoscar[nixl]>=0.11.1`` in each model environment and use reachable worker addresses. Omit ``xavier_cache_bytes`` and retained GPU history settings. Transfer failures raise errors without CPU staging or full-prefill fallback. Relaunch the model after a transfer-worker restart. Directory counters ``gpu_bytes`` and ``imported_tokens`` confirm completed imports; ``host_bytes`` stays zero.

Xavier loads KV asynchronously while other requests continue decoding. Cancelled decode requests retain their destination slots until all writes finish. Each transfer buffer is bounded by the maximum request KV size, with a 512 MiB target and a minimum of one KV page. Reserve GPU memory for both transfer buffers in addition to model weights and TRT's KV pool.

MLX
~~~

.. _mlx-with-xavier:

.. _user_guide_pd_mlx_xavier:

Xavier
^^^^^^

.. versionadded:: v4.0.0

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

Use the default ``xavier`` transport for MLX; native NIXL is unsupported. All workers must reach the supervisor's actor address. Relaunch after a worker restart. Two replicas on one Mac share its Metal GPU; measure your workload before expecting a throughput benefit.

.. _user_guide_pd_cross_engine:

Cross-engine P/D
----------------

.. _cross-engine-vllm-sglang-p-d:

Cross-engine vLLM/SGLang/TRT-LLM P/D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``model_engine`` and ``engine_config`` on each ``replica_config`` entry to select vLLM, SGLang or TRT-LLM independently for each role. Cross-engine P/D requires Xavier GPU transport. Engine options inherit launch arguments; ``engine_config`` overrides them per replica.

.. versionadded:: v4.0.0

   vLLM and SGLang in either prefill/decode direction.

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

For vLLM/SGLang pairings, both workers require identical local weights, tokenizer assets and context limits. Supported models are unquantized FP16 full-attention Qwen2, Qwen3 and Llama text models with TP=PP=DP=1. Requires vLLM >=0.21.0, SGLang >=0.5.21 and ``xoscar[nixl]>=0.11.1``. Xavier checks contracts and prompt token IDs, converts 64-token KV pages on GPU and raises errors on incompatibility or transfer failure.

Streaming and non-streaming requests require ``n=1``. SGLang decode uses the first output token sampled by prefill. vLLM decode imports all but the final prompt token, computes that token and samples the output. Engine kernels can produce small numerical differences. vLLM decode requires at least two prompt tokens.

Cross-engine P/D does not support CPU fallback, retained Xavier GPU history, logprobs, structured sampling, LoRA, speculative decoding, multimodal inputs or hybrid attention. Omit ``xavier_gpu_cache_bytes`` or set it to ``0``; do not set ``xavier_cache_bytes``. Relaunch after a worker restart. Native NIXL comparisons use the same engine on both roles.

.. _cross-engine-trt-llm-p-d:

.. versionadded:: v4.1.0

   TRT-LLM with vLLM or SGLang in either prefill/decode direction.

Xavier supports TRT-LLM prefill with vLLM or SGLang decode, and the reverse directions. All replicas use identical BF16 dense Qwen3 checkpoints, tokenizer assets and context limits, with one CUDA GPU per replica. TRT-LLM requires version ``1.2.1``; vLLM and SGLang requirements match the cross-engine section above. For CUDA/Metal deployments, see :ref:`NVIDIA and MLX cross-engine P/D <trt-mlx-pd>` below.

Set each replica's ``model_engine`` and ``engine_config`` independently. A topology containing TRT-LLM selects BF16 weights and KV for every replica; explicit FP16 options are rejected. Put engine-specific options in ``engine_config`` so they do not reach the other engine.

.. code-block:: python

   client.launch_model(
       model_uid="qwen3-trt-vllm-pd",
       model_name="qwen3",
       model_size_in_billions="0_6",
       model_engine="TRT-LLM",
       model_format="pytorch",
       quantization="none",
       model_path="/models/Qwen3-0.6B",
       replica=2,
       transfer_backend_type="xavier",
       replica_config=[
           {
               "role": "prefill",
               "model_engine": "TRT-LLM",
               "engine_config": {"max_seq_len": 4096},
               "devices": [{"worker_ip": "worker-a:9978", "gpu_idx": [0]}],
           },
           {
               "role": "decode",
               "model_engine": "vLLM",
               "engine_config": {"max_model_len": 4096},
               "devices": [{"worker_ip": "worker-b:9978", "gpu_idx": [0]}],
           },
       ],
   )

For SGLang, use ``model_engine="SGLang"`` and ``context_length`` instead of ``max_model_len``. Reverse the replica roles to reverse the transfer direction. Xavier converts TRT's head-major and the other engines' token-major KV layouts on GPU, preserving BF16 values, then transfers through NIXL. SGLang decode imports full prompt KV and the first sampled token; vLLM and TRT decode replay the final prompt token. Source and destination slots remain owned until GPU operations drain, including cancellation.

Cross-engine sampling restrictions above still apply. TRT additionally does not support ``cache_salt`` or tool calling through this adapter. Retained Xavier history and full-prefill fallback are disabled. CUDA-to-CUDA transfers use NIXL without CPU staging; ``gpu_bytes`` and ``imported_tokens`` measure completed imports and ``host_bytes`` stays zero.

NVIDIA and MLX cross-engine P/D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: v4.0.0

   vLLM/SGLang with MLX in either prefill/decode direction.

Xavier supports both vLLM/SGLang prefill with MLX decode and MLX prefill with vLLM/SGLang decode, with streaming and non-streaming requests. FP16 64-token pages pass through CPU staging and actor RPC between CUDA and Metal. vLLM and MLX decode compute the last prompt token; SGLang decode imports the full prompt KV and the first output token sampled by P. MLX prefill publishes request-owned pages on its worker with a 512 MiB in-flight budget.

Use identical unquantized checkpoint and tokenizer files; do not convert the Mac checkpoint. Worker-local ``model_path`` and ``model_format`` belong in each replica's ``engine_config``. Xavier checks weights, tokenizer, effective model semantics, context limits and prompt token IDs. Model and sampling restrictions match vLLM/SGLang cross-engine P/D; failures raise errors.

.. code-block:: python

   client.launch_model(
       model_uid="qwen-nvidia-mac-pd",
       model_name="qwen2.5-instruct",
       model_size_in_billions="0_5",
       model_engine="vLLM",
       model_format="pytorch",
       quantization="none",
       replica=2,
       replica_config=[
           {
               "role": "prefill", "model_engine": "vLLM",
               "engine_config": {"model_path": "/models/Qwen2.5-0.5B-Instruct",
                                 "max_model_len": 8192},
               "devices": [{"worker_ip": "NVIDIA_WORKER:PORT",
                            "n_gpu": 1, "gpu_idx": [0]}],
           },
           {
               "role": "decode", "model_engine": "MLX",
               "engine_config": {"model_path": "/Users/me/models/Qwen2.5-0.5B-Instruct",
                                 "model_format": "mlx", "context_length": 8192},
               "devices": [{"worker_ip": "MAC_WORKER:PORT", "n_gpu": "auto"}],
           },
       ],
   )

To use MLX prefill, swap the two ``role`` values in the example. For SGLang on either role, use ``SGLang`` and replace ``max_model_len`` with ``context_length``. NVIDIA requires ``xoscar[nixl]>=0.11.1``; Mac requires ``xoscar>=0.11.1`` and ``mlx-lm>=0.31.2``. Workers must reach the supervisor and each other. Use Xavier transport, and omit ``xavier_gpu_cache_bytes`` or set it to ``0``. Relaunch after a worker restart.

For MLX prefill, ``xavier_cache_bytes`` sets the worker-local in-flight CPU page budget (default: 512 MiB). Capacity is reserved before Metal prefill, rounding the full prompt up to 64-token pages. A prompt exceeding the budget fails before prefill; concurrent requests wait for capacity. Reservations and transfer waits are bounded by ``XINFERENCE_SGLANG_XAVIER_TRANSFER_TIMEOUT`` (default: 600 seconds).

MLX-prefill cross-engine requests do not accept ``min_p``, ``presence_penalty``, ``frequency_penalty`` or ``seed``. The router rejects these parameters before scheduling either replica, so the first output token uses the same supported sampling settings as the rest of the response.

Run the supervisor and all worker/model environments with the same Python version and matching ``cloudpickle`` / ``tblib`` versions; actor RPC also carries exceptions and generators.

Compare end-to-end TTFT with the standalone decode engine on its own hardware, using the same effective FP16 weights and cache policy. Network transfers and CUDA/Metal copies may outweigh faster prefill. ``host_bytes`` measures CPU-staged payloads; ``imported_tokens`` records completed KV imports. Correct output alone does not prove reuse.

.. _trt-mlx-pd:
.. _trt-llm-and-mlx-over-a-lan:

.. versionadded:: v4.1.0

   TRT-LLM with MLX in either prefill/decode direction.

Xavier supports TRT-LLM prefill with MLX decode and MLX prefill with TRT-LLM decode across NVIDIA and Apple silicon workers. Use identical, unconverted BF16 dense Qwen3 checkpoints and tokenizer files on both hosts. Every replica uses BF16 weights and KV; FP16 overrides are rejected. The MLX replica automatically selects the unquantized BF16 registry entry. Both decode engines import the prefix and replay the last prompt token.

.. code-block:: python

   client.launch_model(
       model_uid="qwen3-trt-mlx-pd",
       model_name="qwen3",
       model_size_in_billions="0_6",
       model_engine="TRT-LLM",
       model_format="pytorch",
       quantization="none",
       download_hub="huggingface",
       replica=2,
       transfer_backend_type="xavier",
       replica_config=[
           {
               "role": "prefill", "model_engine": "TRT-LLM",
               "engine_config": {"model_path": "/models/Qwen3-0.6B",
                                 "max_seq_len": 4096},
               "devices": [{"worker_ip": "192.168.50.51:9978", "n_gpu": 1}],
           },
           {
               "role": "decode", "model_engine": "MLX",
               "engine_config": {"model_path": "/Users/me/models/Qwen3-0.6B",
                                 "model_format": "mlx", "context_length": 4096},
               "devices": [{"worker_ip": "192.168.50.120:9978", "n_gpu": "auto"}],
           },
       ],
   )

Swap the two roles for MLX prefill. Start workers with LAN-reachable addresses and ensure both workers and the supervisor can reach each other's actor ports. Specify the same ``download_hub`` explicitly. The Python and RPC dependency requirements and MLX-prefill memory budget above also apply. TRT-LLM requires ``1.2.1`` with one CUDA GPU; MLX requires Apple silicon and ``mlx-lm>=0.31.2``.

CUDA/Metal handoff uses bounded host-memory pages and actor RPC over the network. BF16 values travel as raw bytes without FP16 conversion. ``host_bytes`` and ``imported_tokens`` record completed imports; ``gpu_bytes`` stays zero for this transport. Streaming, cancellation and disconnect cleanup are supported. The TRT and MLX sampling restrictions above still apply. Measure TTFT against standalone decode on its own hardware, because network and device copies add latency.
