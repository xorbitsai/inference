.. _user_guide_xavier:
.. _user_guide_vllm_enhancement:

########################
Xavier: KV Cache Sharing
########################
Xavier enables KV cache sharing between model replicas, with adapters for
vLLM, SGLang, TRT-LLM and MLX. Replicas of the same engine can reuse compatible
prefix caches to reduce repeated prefill computation for long prompts and
multi-turn conversations.

Choose a deployment mode
------------------------

Use ordinary replicas to share compatible prefixes while each replica handles
complete requests. By default, shared prefix caching uses replicas of the same
engine. To mix engines in ordinary replicas, see :ref:`xavier-cross-engine-sharing`. For separate prefill and decode roles, including cross-engine and
CUDA/Metal deployments, use :ref:`user_guide_pd_separation`. P/D transport and
ordinary cache storage have separate configuration and compatibility limits.

Usage
-----

Launch multiple replicas with ``enable_xavier=True`` and select a supported ``model_engine``. See :ref:`user_guide_backends` for engine requirements and :ref:`user_guide_pd_separation` for P/D deployment.

The example launches two ordinary replicas. Choose the engine installed on your
workers; the model format and quantization must match it. Each CUDA replica
uses one GPU, while MLX runs on Apple silicon.

.. code-block:: python

   from xinference.client import Client

   client = Client("http://192.168.1.10:9997")
   engine = "vLLM"  # vLLM, SGLang, TRT-LLM or MLX
   model_format, quantization = {
       "vLLM": ("pytorch", "none"),
       "SGLang": ("pytorch", "none"),
       "TRT-LLM": ("pytorch", "none"),
       "MLX": ("mlx", "bf16"),
   }[engine]
   model_uid = client.launch_model(
       model_name="qwen3",
       model_size_in_billions="0_6",
       model_engine=engine,
       model_format=model_format,
       quantization=quantization,
       replica=2,
       n_gpu="auto" if engine == "MLX" else 1,
       context_length=4096,
       enable_xavier=True,
       enable_virtual_env=False,
   )

Due to the underlying communication not recognizing ``0.0.0.0``, the actual IP address needs to be passed when starting Xinference, for example: ``xinference-local -H 192.168.xx.xx``.

Engine requirements
-------------------

.. _xavier-vllm:
.. _limitations:

vLLM
~~~~

.. versionadded:: v1.2.0

V1 connector
^^^^^^^^^^^^

.. versionadded:: v3.4.0

The V1 connector requires vLLM >= ``0.21.0``, TP=1 and PP=1, with text-only
models and no LoRA. Multimodal and prompt-embedding inputs are unsupported.
Versions 0.11 through 0.20 are unsupported. Use the P/D guide for recurrent
model requirements and GPU history settings.

The vLLM GPU transfer path requires NVIDIA GPUs.

.. versionchanged:: v4.0.0

   For attention-only models on vLLM V1, Xavier preserves the model's ``enforce_eager`` setting, allowing CUDA Graph execution. Set ``enforce_eager=True`` at model launch to use eager execution. Recurrent models continue to use eager execution by default.

.. versionadded:: v4.0.0

   Retained vLLM GPU history, configured with ``xavier_gpu_cache_bytes``.

Legacy V0 adapter
^^^^^^^^^^^^^^^^^

The legacy adapter supports vLLM >= ``0.7.0`` and < ``0.11.0`` for ordinary
replica cache sharing. Use the V1 connector for current P/D deployments.

SGLang replicas
~~~~~~~~~~~~~~~

.. versionadded:: v4.0.0

SGLang >= ``0.5.21`` can share prefix KV cache through Xavier's HiCache storage backend. Launch multiple replicas with ``model_engine="SGLang"`` and ``enable_xavier=True``. The initial adapter supports unquantized FP16 Qwen2, Qwen3 and Llama text models with full attention and TP=PP=DP=1; LoRA, speculation, hybrid attention and multimodal inputs are unsupported.

Xavier loads BF16 checkpoints as FP16 and emits a warning. Both weights and KV caches use FP16.

``xavier_cache_bytes`` sets the deployment's shared CPU cache budget (default: ``536870912``, 512 MiB). Each SGLang replica also allocates its own GPU and HiCache host pools. Eviction or unavailable storage becomes a cache miss for ordinary replicas. Cache namespaces include the actual weights, tokenizer, model configuration, KV geometry and engine format; for cross-engine reuse, use :ref:`xavier-cross-engine-sharing`.

Ordinary SGLang replicas share CPU HiCache pages. Explicit prefill/decode roles use Xavier GPU-to-GPU transfer instead. For same-engine or cross-engine P/D and per-replica engine settings, see :ref:`user_guide_pd_separation`.

.. _xavier-trt-llm:

TRT-LLM replicas
~~~~~~~~~~~~~~~~

.. versionadded:: v4.1.0

TRT-LLM requires Linux/CUDA and ``tensorrt-llm==1.2.1``. Xavier supports
unquantized BF16 dense Qwen3 text models, one worker and one GPU per replica
(TP=PP=CP=1). Chunked prefill and overlap scheduling are disabled. Quantization,
hybrid attention, multimodal input, LoRA and speculative decoding are unsupported
with Xavier. See :ref:`trt_llm_backend` for the engine environment requirements.

Ordinary TRT replicas publish complete 64-token KV pages to a shared CPU cache
actor. ``xavier_cache_bytes`` sets the deployment budget (default: 512 MiB).
Partial pages and the final prompt token are computed locally; misses fall back
to local prefill. Model weights, engine KV pools and transfer buffers consume
additional memory. Xavier P/D instead transfers live KV pages through NIXL;
see :ref:`user_guide_pd_separation`.

MLX replicas
~~~~~~~~~~~~

.. versionadded:: v4.0.0

MLX Xavier requires Apple silicon and ``mlx-lm>=0.31.2``. It supports unquantized, full-attention Qwen2, Qwen3 and Llama text models with one worker per replica. Weights and KV cache use FP16. Quantized weights or KV cache, rotating caches, hybrid attention, multimodal inputs, LoRA and speculative decoding are unsupported.

``xavier_cache_bytes`` bounds the shared CPU cache (default: 512 MiB). Each replica also owns its model and Metal KV cache. Ordinary replicas compute locally on a miss or unavailable storage; incompatible deployments cannot share pages. Returned usage reports reused tokens in ``prompt_tokens_details.cached_tokens``. Explicit prefill/decode roles enable MLX Xavier P/D automatically. For cross-engine reuse, use :ref:`xavier-cross-engine-sharing`.

.. _xavier-cross-engine-sharing:

Cross-engine prefix sharing
---------------------------

.. versionadded:: v4.2.0

Ordinary replicas can share cached prefixes across vLLM, SGLang, TRT-LLM and MLX. Each replica still handles complete requests: a later request on SGLang can reuse a prefix computed by vLLM, and vice versa. Enable Xavier and select an engine for each replica; no prefill/decode roles are needed.

.. code-block:: python

   model_uid = client.launch_model(
       model_uid="shared-prefix",
       model_name="qwen3",
       model_size_in_billions="0_6",
       model_engine="vLLM",
       model_format="pytorch",
       quantization="none",
       replica=2,
       enable_xavier=True,
       xavier_cache_bytes=536870912,
       replica_config=[
           {
               "model_engine": "vLLM",
               "engine_config": {"max_model_len": 4096},
               "devices": [{"worker_ip": "CUDA_WORKER_ADDRESS", "n_gpu": 1,
                            "gpu_idx": [0]}],
           },
           {
               "model_engine": "SGLang",
               "engine_config": {"context_length": 4096},
               "devices": [{"worker_ip": "CUDA_WORKER_ADDRESS", "n_gpu": 1,
                            "gpu_idx": [1]}],
           },
       ],
   )

All replicas must use identical local checkpoint and tokenizer files, effective context limits and unquantized full-attention models. vLLM/SGLang/MLX combinations support FP16 Qwen2, Qwen3 and Llama text models. A topology containing TRT-LLM requires BF16 dense Qwen3 on every replica. Use one worker and one device per replica; existing engine version requirements apply.

For MLX, set its replica's ``engine_config`` to include ``model_format="mlx"`` and its local ``model_path``. Use the original unconverted checkpoint on the Mac. Engine-specific options belong in ``engine_config``; worker-local paths may also be set there for CUDA replicas. Configure the same effective context limit for every engine.

Xavier stores immutable, complete 64-token pages in a supervisor-owned CPU cache and transfers them through actor RPC. ``xavier_cache_bytes`` bounds this shared pool (default: 512 MiB), excluding model weights, engine caches and in-flight copies. Partial tails and the final prompt token are computed locally. Misses, eviction and unavailable storage fall back to local prefill. Incompatible models fail configuration before sharing pages.

This mode uses the native backend. LMCache/Mooncake storage and retained GPU history are not supported for cross-engine sharing. Quantization, sharded KV, LoRA, speculation, multimodal input and hybrid attention remain unsupported. TRT-LLM retains its existing ``cache_salt`` restriction; vLLM, SGLang and MLX preserve request-scoped isolation.

Requests must use options supported by every engine in the deployment. TRT-LLM rejects tools, ``response_format``, log probabilities, ``echo`` and ``n>1``. MLX Xavier rejects structured ``response_format``, forced tool choice and thinking budgets. Otherwise, success can depend on which replica receives the request.

Measure repeated-prefix TTFT against local prefill on the receiving engine. CPU copies and network transfer can exceed prefill cost for short prefixes. Use cached-token usage to confirm reuse on every engine. The ``imported_pages`` counter reports completed imports only for vLLM and TRT-LLM receivers; SGLang and MLX receivers use cached-token usage instead. Response correctness alone does not prove a cache hit.

Storage backends
----------------

Native storage
~~~~~~~~~~~~~~

The default ``xavier_cache_backend="native"`` uses each engine's Xavier cache
implementation. ``xavier_cache_bytes`` sets the shared CPU budget for SGLang,
TRT-LLM and MLX; it is not a vLLM setting. vLLM's GPU history uses
``xavier_gpu_cache_bytes``. These budgets exclude weights, engine KV pools and
transfer buffers. Choose the engine above before sizing storage.

LMCache
~~~~~~~

.. versionadded:: v4.1.0

LMCache storage for ordinary vLLM V1, SGLang and TRT-LLM replicas.

``xavier_cache_backend="native"`` is the default and keeps the existing CPU and GPU cache behavior. The optional ``lmcache`` backend adds external prefix-cache storage for ordinary vLLM V1, SGLang and TRT-LLM replicas. Install ``xinference[xavier-lmcache]`` in the Linux environment hosting storage: each vLLM transfer actor's environment, or the supervisor environment for SGLang and TRT-LLM. Model environments can remain separate. This adapter targets LMCache >= ``0.5.6`` and < ``0.6`` with the native extension for the selected L2 backend, and Python 3.10 through 3.13.

Select the backend when launching the model:

.. code-block:: python

   model_uid = client.launch_model(
       model_name="qwen3",
       model_engine="vLLM",
       model_format="pytorch",
       model_size_in_billions=8,
       quantization="none",
       replica=2,
       enable_xavier=True,
       xavier_cache_backend="lmcache",
       xavier_cache_backend_config={
           "cpu_bytes": 536870912,
           "disk_path": "/mnt/nvme/xavier",
           "disk_bytes": 10737418240,
           "read_timeout": 30.0,
       },
   )

``cpu_bytes`` bounds each LMCache instance's DRAM pool (default: 512 MiB). vLLM creates one instance per replica; SGLang and TRT-LLM share one supervisor-owned instance per deployment. ``disk_path`` must be absolute and local to that storage host, and ``disk_bytes`` is the positive disk capacity per instance. LMCache manages asynchronous writes, prefetch and eviction within these tiers. Disk eviction uses watermarks, so the configured disk capacity is not a filesystem quota. Xavier also uses bounded transient CPU staging and read copies, up to another ``2 * cpu_bytes`` per instance; engine KV pools, owned import buffers, serialization and transfer buffers consume additional memory.

``read_timeout`` bounds a cache lookup's wait in seconds (default: 30). Missing, unavailable or timed-out cache reads fall back to local prefill. At most one LMCache prefetch job runs per storage instance; a timed-out or cancelled job still releases its LMCache read locks. Reads exceeding the transient capacity return a bounded prefix or fall back to local prefill. Compare TTFT and read latency against prefill cost for your workload.

For ordinary replicas, the LMCache backend requires multiple replicas of the same engine and preserves that engine's existing Xavier model, dtype and parallelism limits; legacy vLLM V0 and ordinary MLX replicas are unsupported. For P/D roles, see :ref:`xavier-pd-lmcache`. Use backend ``cpu_bytes`` instead of the native ``xavier_cache_bytes`` or ``xavier_gpu_cache_bytes`` budgets. Backend selection does not change cache-salt behavior. Each deployment and storage instance owns a separate namespace and directory; relaunching does not recover old entries. Graceful shutdown removes that directory; after a process crash, remove abandoned deployment directories when they are no longer in use.

.. _xavier-pd-lmcache:

LMCache history for P/D
^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: v4.2.0

Select ``xavier_cache_backend="lmcache"`` and pass the backend configuration above when launching Xavier P/D roles. Prefill can run on vLLM, SGLang, TRT-LLM or MLX, including cross-engine deployments. Decode keeps the existing Xavier handoff and does not access LMCache. Native NIXL deployments cannot use this option.

P restores compatible historical prefix pages before computing the remaining prompt. Owned CPU snapshots enter a worker-local write queue bounded by ``cpu_bytes`` per P replica; LMCache writes run in the background. A full queue or failed write drops optional history. Missing or failed reads compute locally, while live P/D transfer failures still raise errors. CPU snapshot packing and local enqueueing remain part of prefill latency; asynchronous writes do not remove those costs.

One shared LMCache history store runs on the first CUDA worker; install ``xinference[xavier-lmcache]`` in that worker's environment. All-MLX groups require a Linux model-service host for the store. Each P replica adds a bounded CPU write queue and owned read buffers; SGLang also allocates its native HiCache host pool. Include these allocations in addition to the LMCache memory budgets above. Disk paths belong to the storage host.

For MLX prefill, ``xavier_cache_bytes`` still bounds live host handoff pages independently of LMCache. CUDA P/D may omit ``xavier_gpu_cache_bytes`` or set it to ``0``; retained Xavier GPU history is disabled with this backend.

This option uses the canonical P/D model contract and its model, dtype, single-device and 64-token page restrictions, even for same-engine CUDA groups. See :ref:`user_guide_pd_separation`. Engine-local prefix caching remains available on P. Different cache salts remain isolated; TRT-LLM's existing rejection of ``cache_salt`` is unchanged. Historical entries are scoped to the deployment and are not recovered after relaunch.

Mooncake Store
^^^^^^^^^^^^^^

.. versionadded:: v4.2.0

Set ``l2_backend="mooncake"`` in ``xavier_cache_backend_config`` to use LMCache's Mooncake Store adapter. The default ``l2_backend="filesystem"`` preserves the disk configuration above. This selects storage for ordinary prefix-cache reuse or P-side history; P/D transport and supported engine/model combinations remain the same.

The storage environment needs LMCache built with ``BUILD_MOONCAKE=1`` and compatible Mooncake SDK headers and libraries. Start the Mooncake master, metadata service and storage pool first. Use a reachable ``local_hostname`` without a port; Mooncake allocates a distinct port for each storage actor. Replace the endpoints in this TCP example with reachable service addresses, and pass the object as ``xavier_cache_backend_config``:

.. code-block:: json

   {
     "l2_backend": "mooncake",
     "cpu_bytes": 536870912,
     "storage_bytes": 10737418240,
     "read_timeout": 30,
     "mooncake_config": {
       "local_hostname": "storage-node",
       "metadata_server": "http://store-host:18080/metadata",
       "master_server_addr": "store-host:15051",
       "protocol": "tcp",
       "global_segment_size": 0,
       "local_buffer_size": 67108864
     }
   }

``storage_bytes`` bounds this actor's tracked remote cache capacity, together with ``cpu_bytes``; it is not a Mooncake cluster quota. At capacity, a new page is skipped and one least-recently-used owned object is scheduled for deletion. Only one deletion runs at a time, after outstanding LMCache writes drain; sustained traffic with new prefixes can therefore skip many new pages. Mooncake manages actual storage capacity and eviction; ``global_segment_size`` is the memory this client contributes. ``disk_path`` and ``disk_bytes`` apply only to filesystem storage and are rejected with Mooncake. ``protocol="rdma"`` additionally requires working RDMA hardware and drivers; LMCache registers the eagerly allocated L1 buffer.

Deployment, schema and cache-salt isolation still apply. vLLM uses a distinct storage namespace per replica rank; SGLang and TRT-LLM replicas share one supervisor-owned storage actor. Other deployments' objects are not discovered, and relaunching does not recover old entries. Cache misses, unavailable reads and timeouts fall back to local prefill. Graceful shutdown drains writes and deletes only this actor's tracked namespaced objects before closing its client. If cleanup fails or the process crashes, remaining objects are subject to Mooncake's eviction policy. Cleanup never explicitly deletes other clients' keys or stops the shared service. A client contributing memory withdraws that segment on close; use ``global_segment_size=0`` with a separate storage pool to avoid this.

.. _xavier-cache-isolation:

Request-scoped cache isolation
------------------------------

.. versionadded:: v4.1.0

Pass ``cache_salt`` with each inference request to isolate prefix-cache reuse. Requests with the same salt can reuse compatible caches across replicas; different salts cannot reuse each other's cached prefixes. This covers the local engine cache, Xavier shared caches and retained history, and same-engine or cross-engine P/D for vLLM, SGLang and MLX. Both P/D roles receive the same salt; incompatible cache scopes are rejected before importing KV.

With the Xinference Python client, include ``cache_salt`` in ``generate_config``. With the OpenAI client, use ``extra_body``:

.. code-block:: python

   client.chat.completions.create(
       model="my-model",
       messages=[{"role": "user", "content": "Hello"}],
       extra_body={"cache_salt": customer_cache_salt},
   )

The same top-level field is accepted by the Completions, Chat Completions, Responses and Anthropic Messages endpoints, for streaming and non-streaming inference. A salt must be a nonempty string of at most 256 characters without control characters. Unsupported model engines reject salted requests rather than ignoring the field. Legacy vLLM V0 requires Xavier to use salts.

Choose a stable, unguessable salt for each intended sharing boundary. A trusted application or gateway should authenticate callers, derive the salt from its own customer identity, and overwrite any client-supplied salt. Xinference does not map salts to users or create a tenant database. Salt knowledge permits cache sharing within that scope; client-selected salts alone do not enforce customer identity.

Without a salt, requests keep the existing shared-cache behavior. Set ``XINFERENCE_REQUIRE_CACHE_SALT=1`` in the API server environment to reject inference requests without a salt; the default is ``0``. Enable this policy after all callers or an API extension supply salts. The policy applies to the four inference endpoints above and does not disable prefix caching.

Distribution wrappers can supply an authoritative salt through the optional synchronous ``cache_salt(request)`` extension hook, returning ``xinference.extensions.CacheSalt(value)``. The hook runs after model-access checks and may read identity prepared by authenticated API dependencies from ``request.state``. Its result overrides the request field. Configure the extension through ``XINFERENCE_EXTENSIONS``; provider failures, missing return values and multiple providers reject the request without falling back to the client salt. When no provider is configured, the request field is used. Extension hooks must remain non-blocking.

Cache salting limits prefix-cache timing inference across sharing boundaries. It does not provide separate model processes, memory quotas or complete timing-side-channel isolation. Changing a salt starts a new cache scope; existing entries remain subject to normal eviction.

TRT-LLM does not support ``cache_salt`` through this adapter.
