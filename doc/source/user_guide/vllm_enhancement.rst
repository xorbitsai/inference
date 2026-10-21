.. _user_guide_vllm_enhancement:

############################################
Xavier: Share KV Cache between vllm replicas
############################################
For scenarios such as long document queries and multi-round conversations,
the computation during the inference prefill phase can be particularly heavy,
which affects overall throughput and the latency of individual inferences.
Xinference enhances the vllm engine by introducing the ``Xavier`` framework,
enabling KV cache sharing across multiple vllm instances.
This allows KV cache computed by other replicas to be directly reused, avoiding redundant computations.

*****
Usage
*****
Simply add the parameter ``enable_xavier=True`` when starting the vllm model.

.. versionchanged:: v4.0.0

   For attention-only models on vLLM V1, Xavier preserves the model's ``enforce_eager`` setting, allowing CUDA Graph execution. Set ``enforce_eager=True`` at model launch to use eager execution. Recurrent models continue to use eager execution by default.

***********
Limitations
***********
* Xavier retains the V0 adapter for vLLM >= ``0.7.0`` and < ``0.11.0``.
  The V1 connector supports vLLM >= ``0.21.0``. See :ref:`user_guide_pd_separation`
  for community Prefill/Decode deployment.
* Due to the underlying communication not recognizing ``0.0.0.0``, the actual IP address needs to be passed when starting Xinference, for example: ``xinference-local -H 192.168.xx.xx``.
* The vLLM GPU transfer path requires NVIDIA GPUs.

Storage backends
================

``xavier_cache_backend="native"`` is the default and keeps the existing CPU and GPU cache behavior. The optional ``lmcache`` backend adds disk-backed prefix reuse for ordinary vLLM V1 replicas on Linux NVIDIA workers. Install ``xinference[xavier-lmcache]`` in each model environment; this adapter targets LMCache >= ``0.5.6`` and < ``0.6`` with its native FS extension, and Python 3.10 through 3.13.

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

``cpu_bytes`` bounds each replica's LMCache DRAM pool (default: 512 MiB). ``disk_path`` must be an absolute worker-local path, and ``disk_bytes`` is the positive disk capacity per replica. LMCache manages asynchronous writes, prefetch and eviction within these tiers. Disk eviction uses watermarks, so the configured disk capacity is not a filesystem quota. Xavier also uses bounded transient CPU staging and read copies, up to another ``2 * cpu_bytes`` per replica; engine KV pools, serialization and transfer buffers consume additional memory.

``read_timeout`` bounds a cache lookup's wait in seconds (default: 30). Missing, unavailable or timed-out cache reads fall back to local prefill. At most one LMCache prefetch job runs per replica; a timed-out or cancelled job still releases its LMCache read locks. Prefixes exceeding the transient read capacity are recomputed locally. Compare TTFT and read latency against prefill cost for your workload.

This initial LMCache adapter requires multiple ordinary replicas and the existing Xavier V1 compatibility limits. P/D roles, explicit ``xavier_gpu_cache_bytes``, legacy vLLM V0, SGLang and MLX are unsupported. Cache keys retain request salt isolation. Each launch and replica owns a separate storage namespace and directory; relaunching does not recover old cache entries. Graceful shutdown removes that directory; after a process crash, remove abandoned deployment directories when they are no longer in use.

SGLang replicas
===============

.. versionadded:: v4.0.0

SGLang >= ``0.5.21`` can share prefix KV cache through Xavier's HiCache storage backend. Launch multiple replicas with ``model_engine="SGLang"`` and ``enable_xavier=True``. The initial adapter supports unquantized FP16 Qwen2, Qwen3 and Llama text models with full attention and TP=PP=DP=1; LoRA, speculation, hybrid attention and multimodal inputs are unsupported.

Xavier loads BF16 checkpoints as FP16 and emits a warning. Both weights and KV caches use FP16.

``xavier_cache_bytes`` sets the deployment's shared CPU cache budget (default: ``536870912``, 512 MiB). Each SGLang replica also allocates its own GPU and HiCache host pools. Eviction or unavailable storage becomes a cache miss for ordinary replicas. Cache namespaces include the actual weights, tokenizer, model configuration, KV geometry and engine format; caches cannot yet be exchanged between vLLM and SGLang.

Ordinary SGLang replicas share CPU HiCache pages. Explicit prefill/decode roles use Xavier GPU-to-GPU transfer instead. For same-engine or cross-engine P/D and per-replica engine settings, see :ref:`user_guide_pd_separation`.

MLX replicas
============

.. versionadded:: v4.0.0

MLX replicas can share prefix KV cache with ``model_engine="MLX"`` and ``enable_xavier=True``. The supported models and FP16 requirements are described in :ref:`user_guide_pd_separation`. ``xavier_cache_bytes`` bounds the shared CPU cache (default: 512 MiB). Each replica also owns its model and Metal KV cache. Ordinary replicas compute locally on a miss or unavailable storage; incompatible deployments cannot share pages. Returned usage reports reused tokens in ``prompt_tokens_details.cached_tokens``. Explicit prefill/decode roles enable MLX Xavier P/D automatically. Cross-engine cache sharing is not yet supported.

Request-scoped cache isolation
==============================

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
