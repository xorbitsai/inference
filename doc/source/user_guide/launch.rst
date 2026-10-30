.. _launch:

============================
Model Launching Instructions
============================

This document aims to provide a functional overview of model launching.

On **Running Models**, choose **Clear failed record** to remove a failed launch after its runtime has been cleaned up. You can also use ``DELETE /v1/models/{model_uid}``. This clears the status record and keeps cached model files. With authentication enabled, the operation requires ``models:write`` permission.

An initial launch remains cancellable until all its replicas finish, even when some are already ready. **Cancel launch** on **Running Models** calls ``POST /v1/models/{model_uid}/cancel`` to stop pending work and remove replicas created by that launch. ``GET /v1/models/instances`` reports ``can_cancel_launch`` for this operation. Scale-up of an established model does not enable initial-launch cancellation.

.. note::

   The LMDeploy backend has been removed because its Xinference integration is
   no longer maintained. For InternVL models, select vLLM or Transformers instead.

Updating the model catalog
==========================

On the **Launch Model** page, choose the model type beside **Update**. **Update** refreshes the whole category. Choose **Update one model** and enter the exact Hub model name (or world family name) to update just that model, including a model not yet listed. Other catalog entries are preserved. This updates shared metadata without downloading weights or restarting running models. With authentication enabled, both actions require administrator or ``models:register`` permission.

Model favorites
===============

Stars on the **Launch Model** page are saved for the current account and survive browser changes and server restarts. Favorites are identified by model name across model categories. Reading or changing favorites requires ``models:list`` permission. Without authentication, all visitors share one collection. With authentication enabled, changes require a user session; API keys can only read.

Favorites use a ``model_favorites`` table in the selected authentication database. With the default SQLite backend, this is ``<XINFERENCE_HOME>/auth/auth.db`` (configured by ``XINFERENCE_AUTH_DB_PATH``). Without authentication, the same path stores the shared collection and only the favorites table is initialized.

Old browser favorites are not imported automatically because they have no account identity. Choose **Import browser favorites** to add them to the current account. The browser copy is retained if any write fails or the account changes during import; retrying does not create duplicates.

``GET /v1/model_favorites`` returns a list of model names for the authenticated account. ``PUT`` and ``DELETE`` on the same path accept ``{"model_name": "org/model"}`` to add or remove one name. Both operations are idempotent. The server derives the owner from authentication; clients cannot choose another owner.

.. _launch_recommendations:

LLM launch recommendations
==========================

.. versionadded:: v3.5.0

In an LLM's deployment dialog, use the launch recommendation button to get a
suggested configuration, then review it before deploying. The button is available
for LLMs, embedding, rerank, and audio models. Recommendations currently target a single replica.
Custom model paths and custom engine parameters require manual configuration.

How recommendations are chosen
------------------------------

Click **Recommend configuration** in the upper-right corner of the launch
dialog to fill in the engine, format, size, and quantization together. You can
still change these fields before clicking **Deploy**. The recommendation also
selects a worker, so the configuration is tied to the machine checked for it.

Xinference starts with your choices. If you have selected an 8B model, it looks
for configurations of that size; it will not switch to a smaller model. Worker,
GPU, and virtual environment settings also constrain the search. It checks each
worker separately and removes combinations that fail the compatibility checks.
If none remain, it leaves your form unchanged so you can adjust the settings.

If memory data is available, Xinference prefers the largest size estimated to
fit. Otherwise it falls back to the smallest unverified size. Within that size,
installed engines are preferred; virtual-environment candidates remain eligible.

Next comes the platform's default engine order. With the earlier conditions
equal, MLX comes first on Apple Silicon, vLLM followed by SGLang on Linux with
CUDA, and llama.cpp on CPU. Within an engine, common 4-bit quantizations such as
Q4_K_M and Int4 are preferred when supported. If choices are still tied, weights
already cached on that worker get priority. A cached model does not override
your selected size or the earlier preferences.

Estimates use offline architecture facts and checkpoint sizes when available.
MoE weights include all resident experts. Full-attention KV cache, Gated DeltaNet
recurrent state and workspace are counted separately. MLX, Transformers, vLLM,
SGLang and llama.cpp share this accounting with different workspace bounds.
Unknown layouts, including MLA, remain unverified.

Defaults are one sequence, the model's registered maximum context, FP16 KV cache, no image tokens and
no retained prefix cache. The budget is 80% of currently free memory: the lowest
free memory among eligible GPUs, or available system memory on CPU and Apple
Silicon. The default vLLM GPU reservation is checked separately when total memory
is known; SGLang budgets from memory available at initialization. Multi-GPU
requests keep compatibility-only selection.
Known oversized candidates are excluded. Workspace bounds are estimates, not
measured peaks. Engine preallocation is not modeled beyond the separate vLLM
default reservation check; custom engine settings require manual configuration.
No resources are reserved.

Using the API
-------------

``POST /v1/models/recommend`` is read-only. Supply ``model_name`` and
``model_type="LLM"``; the optional ``constraints`` object accepts
``model_size_in_billions``, ``worker_ip``, ``enable_virtual_env``, ``n_gpu``, and
``gpu_idx``. LLM constraints also accept ``context_length``. Supplied constraints
are hard requirements, never silently relaxed.

LLM constraints also accept positive integers ``context_length`` (text input plus
output tokens) and ``concurrency``, and non-negative integers ``vision_tokens``
(encoded image tokens per sequence) and ``prefix_cache_bytes`` (additional retained
cache across the instance). Omitting ``context_length`` uses each candidate's
registered maximum context, including encoded image tokens. Models without a
known maximum remain unverified. Recommended vLLM ``max_model_len`` and SGLang
``context_length`` equal text plus vision tokens per sequence. llama.cpp uses
``n_parallel=concurrency`` and ``n_ctx=(context_length + vision_tokens) * concurrency``
because its context is divided across slots. The Web UI applies these advanced
launch parameters. Other workload fields affect estimates rather than request
limits. Older workers that do not support workload options are retried without
them and remain unverified, with a version warning. Image workloads require vision metadata.
For multiple images, ``vision_tokens_per_image`` specifies their maximum encoded
token count; zero assumes all image tokens belong to one image. Cached states grow
with concurrency, while model weights are counted once. CUDA profiles for vLLM and SGLang share a
16,384-token prefill workspace profile with fused attention and sampled-row logits.

Recommendation configs expose ``context_length`` as the total per-request budget,
including encoded image tokens. llama.cpp also returns ``n_parallel`` to match the
estimated concurrency; the launch layer maps the unified budget to each engine.

.. code-block:: bash

    curl -X POST http://127.0.0.1:9997/v1/models/recommend \
      -H 'Content-Type: application/json' \
      -d '{
        "model_name": "qwen2.5-instruct",
        "model_type": "LLM",
        "constraints": {"model_size_in_billions": "7", "n_gpu": 1}
      }'

The response ``status`` is ``recommended`` or ``no_recommendation``. On success,
``config`` contains launch-compatible fields such as ``model_engine``,
``model_format``, ``model_size_in_billions``, ``quantization``, and ``worker_ip``.
``reasons`` and ``warnings`` contain entries with ``code`` and ``message`` fields.
If no configuration satisfies the constraints, review the explanation and adjust
the constraints or configure the launch manually.

A recommendation does not guarantee sufficient memory or a successful launch.
It reserves no resources, downloads no model files, and installs no dependencies.
Existing launch behavior is unchanged; deployment remains a separate action.

``config`` also includes the effective
``enable_virtual_env`` and any supplied ``n_gpu`` and ``gpu_idx`` constraints.
``n_gpu=null`` requests CPU placement, while ``n_gpu="auto"`` uses the existing
launch allocation policy rather than calculating the required GPU count.

Both ``recommended`` and ``no_recommendation`` return HTTP 200. An unknown model
returns HTTP 404; invalid requests return HTTP 422. Reason and warning codes use
lower snake_case. The ``memory_not_verified`` warning is always included.

Unified LLM context length
==========================

Set ``context_length`` in the launch API or use ``--context-length`` in the CLI
to choose the input plus output token limit per request. In the Web UI, use the
context slider or enter an exact integer; **Model default** clears the override.
The slider starts at the model's native context length. For models and engines
with a registered YaRN recipe, its upper bound includes the verified extension
range. Hover over the question mark beside **Context length** to see the native
length, extension limit, and whether automatic YaRN is supported by the selected
engine. Context selections are preserved in configuration history and passed to
the recommendation API.

The launch layer maps this setting to vLLM's ``max_model_len``, SGLang,
Transformers and MLX's ``context_length``, and
llama.cpp's ``n_ctx=context_length*n_parallel`` total slot capacity. Existing engine
parameters still work; contradictory unified and native values are rejected.
Omitting the setting preserves engine defaults.

Qwen3.5, Qwen3.6, and Qwen3.8 have registered automatic YaRN recipes for vLLM and
SGLang; Qwen3.8-Flash-Next currently supports vLLM only. Selecting a length within
the native window leaves RoPE unchanged. Selecting a longer length enables YaRN
at model launch, with the scaling factor rounded up from the requested length
divided by the native length (for example, 524,288 tokens uses a factor of 2 and
1,000,000 tokens uses a factor of 4 for a 262,144-token model). The required
environment flag is applied only to the model process. The automatic limit is
1,010,000 tokens for Qwen3.5, Qwen3.6, and Qwen3.8-Max, and 1,000,000 tokens for
Qwen3.8-27B and Qwen3.8-Flash-Next. Explicit RoPE overrides and already-scaled
checkpoint configurations take precedence. Models without a registered recipe
retain their existing engine behavior. Changing the selected length requires
restarting the model; the engine does not switch scaling per request.

For models with a registered YaRN recipe, weight-cache reload cannot increase
the current context length. Deploy a new model to increase it so the checkpoint
and RoPE settings are checked again.

With an explicit context limit, MLX rejects prompts that leave no room for output
and caps ``max_tokens`` to the remaining budget. Cached prompt tokens and
processed image tokens count toward the input limit. Non-streaming requests that
exceed the input limit return HTTP 400; streaming requests report an SSE error
without starting generation.

.. code-block:: bash

    xinference launch --model-name qwen3.5 --model-engine vLLM \
      --size-in-billions 9 --model-format awq --quantization Int4 \
      --context-length 32768

To launch Qwen3.8-27B with automatic YaRN extension to one million tokens:

.. code-block:: bash

    xinference launch --model-name qwen3.8 --model-engine vLLM \
      --size-in-billions 27 --model-format pytorch --quantization none \
      --context-length 1000000

Recommendations for other model types
======================================

.. versionadded:: v3.5.0

Embedding, rerank, and audio models use the same **Recommend configuration** button and ``POST /v1/models/recommend`` endpoint; set ``model_type`` to ``embedding``, ``rerank``, or ``audio``. The recommendation fills only the fields supported by that type, without selecting a model size. Embedding and rerank prefer an available sentence-transformers engine; audio prefers an available MLX engine on Apple Silicon, otherwise its standard Transformers or PyTorch backend. Installed engines come before engines that need a virtual environment, and unquantized variants are preferred within an engine. Unlike LLM recommendations, these types do not use cached weights as a tie-breaker. Worker, GPU, and virtual environment constraints still apply; ``model_size_in_billions`` is rejected for these types. A recommendation remains a starting configuration, not a memory-fit guarantee. Image and video recommendations are not yet supported.

Download without launching
==========================

.. versionadded:: v3.4.0

Open a model's deployment dialog and choose **Download only** beside the
**Deploy** button. It downloads the same model artifacts used by deployment,
but does not reserve GPUs, create model subprocesses, install a virtual
environment, or load a model into memory.

The REST endpoint is ``POST /v1/cache/models``. Supply ``cache_uid`` when the
client needs to query progress or cancel the operation while the POST request
is still running.

.. code-block:: bash

    curl -X POST http://127.0.0.1:9997/v1/cache/models \
      -H 'Content-Type: application/json' \
      -d '{
        "cache_uid": "download-qwen",
        "model_name": "qwen2.5-instruct",
        "model_type": "LLM",
        "model_engine": "transformers",
        "model_format": "pytorch",
        "model_size_in_billions": "0_5",
        "quantization": "none"
      }'

Progress and cancellation use the same ``cache_uid``:

.. code-block:: bash

    curl http://127.0.0.1:9997/v1/cache/models/download-qwen/progress
    curl -X POST http://127.0.0.1:9997/v1/cache/models/download-qwen/cancel

Cache-only downloads can also be paused and resumed:

.. code-block:: bash

    curl -X POST http://127.0.0.1:9997/v1/downloads/download-qwen/pause
    curl -X POST http://127.0.0.1:9997/v1/downloads/download-qwen/resume

Xinference persists unfinished cache-only download tasks. If the supervisor or
worker stops unexpectedly, an active task is shown as ``interrupted`` after
restart and can be resumed with the same endpoint. Resume reuses files already
present in the model hub cache. Byte-range continuation of a partially written
file depends on the selected hub client; otherwise that incomplete file is
downloaded again.

``GET /v1/downloads`` lists active, paused, interrupted, and failed downloads.
Pause and resume apply to cache-only downloads. A model launch that happens to
be downloading still uses the launch cancellation endpoint because pausing a
launch would also need to preserve its runtime allocation state.

In a distributed deployment, ``worker_ip`` can select the worker whose local
cache receives the files. If it is omitted, the supervisor selects one worker.

``GET /v1/cache/models`` includes ``model_type`` and ``model_engine`` for each cached model. The engine list reports candidates discovered on the worker that owns its files, including that worker's virtual environment setting. Rerank and world cache versions identify only the model name, so their candidates are not checked against a cached format. ``null`` means the metadata is unknown or discovery failed; an empty list means no matching engine is currently discoverable. Listing caches does not download weights, prepare environments, or launch models.

Replica
=======

During deployment, the progress endpoint ``GET /v1/models/{model_uid}/progress``
reports a stage for each replica. After model files finish downloading, a replica
can report ``waiting_for_dependencies`` while another launch prepares its shared
virtual environment, then ``installing_dependencies`` while packages are being
installed, and ``loading`` after the environment is ready.

The Web UI shows each replica's file download details inside its replica card.

During dependency installation, each replica's progress includes the number of
packages planned for the main install and the number already present in its
virtual environment. A replica waiting for a shared environment has no install
count of its own. If the installer cannot resolve a reliable plan, the count is
omitted.

The Web UI keeps the dependency installation result visible while a replica is
loading. When no dependency installation is needed, the replica card says so.

Replicas specify the number of model instances to load. For example, if you have two GPUs and each can host one replica of the model,
you can set the replica count to 2. This way, two identical instances of the model will be distributed across the two GPUs.
Xinference automatically load-balances requests to ensure even distribution across multiple GPUs.
Meanwhile, users see it as a single model, which greatly improves overall resource utilization.

Traditional Multi-Instance Deployment：

When you have multiple GPU cards, each capable of hosting one model instance, you can set the number of instances equal to the number of GPUs. For example:

- 2 GPUs, 2 instances: Each GPU runs one model instance
- 4 GPUs, 4 instances: Each GPU runs one model instance

.. versionadded:: v1.15.0

Introduce a new environment variable:

.. code-block:: bash

    XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU

Control whether to enable the single GPU multi-copy feature
Default value: 1

New Feature: Smart Replica Deployment

1. Single GPU Multi-Replica

New Support: Run multiple model replicas even with just one GPU.

- Scenario: You have 1 GPU with sufficient VRAM
- Configuration: Replica Count = 3, GPU Count = 1
- Result: 3 model instances running on the same GPU, sharing GPU resources

2. Hybrid GPU Allocation

Smart Allocation: Number of replicas may differ from GPU count; system intelligently distributes

- Scenario: You have 2 GPUs and need 3 replicas
- Configuration: Replicas=3, GPUs=2
- Result: GPU0 runs 2 instances, GPU1 runs 1 instance

Per-replica placement
---------------------

For distributed deployments, ``replica_config`` can pin every replica to a
specific worker and optional GPU indexes. Worker addresses must use the full
registered ``IP:port`` value. The number of entries must equal ``replica`` and
each entry currently supports exactly one worker.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model_uid = client.launch_model(
        model_name="qwen2.5-instruct",
        model_engine="vllm",
        replica=2,
        replica_config=[
            {
                "replica_uid": "primary",
                "devices": [
                    {
                        "worker_ip": "192.168.1.10:9978",
                        "n_gpu": 1,
                        "gpu_idx": [0],
                    }
                ],
            },
            {
                "replica_uid": "secondary",
                "devices": [
                    {
                        "worker_ip": "192.168.1.11:9978",
                        "n_gpu": 1,
                        "gpu_idx": [0],
                    }
                ],
            },
        ],
    )

The same placement can be supplied to ``xinference launch`` as a JSON array:

.. code-block:: bash

    xinference launch \
      --model-name qwen2.5-instruct \
      --model-engine vLLM \
      --replica 2 \
      --replica-config '[{"replica_uid":"primary","devices":[{"worker_ip":"192.168.1.10:9978","n_gpu":1,"gpu_idx":[0]}]},{"replica_uid":"secondary","devices":[{"worker_ip":"192.168.1.11:9978","n_gpu":1,"gpu_idx":[0]}]}]'

``--replica_config`` remains available as a compatibility alias, but
``--replica-config`` is recommended. If ``--replica`` is omitted, the CLI
derives it from the number of array entries. If it is supplied explicitly, it
must equal that number.

``replica_config`` is mutually exclusive with the model-level ``worker_ip``,
``n_gpu``, and ``gpu_idx`` arguments, and with ``n_worker > 1``. The CLI
therefore rejects ``--replica-config`` together with ``--worker-ip``,
``--gpu-idx``, a non-``auto`` ``--n-gpu``, or ``--n-worker`` greater than 1.
Each replica currently targets exactly one worker. If ``replica_uid`` is
omitted, Xinference assigns the stable default
``{model_uid}-{replica_index}`` (for example, ``my-model-0``). Omit ``gpu_idx``
and use ``n_gpu="auto"`` to let the selected worker allocate GPUs
automatically.

Running models can be scaled by one or more replicas in a single operation.
The new replicas reuse the existing launch configuration by default. You can
override their model engine and device allocation without changing existing
replicas. Placement is optional; if omitted, the supervisor selects a worker
automatically.

.. code-block:: python

    result = client.add_model_replica(model_uid)

    scale_result = client.add_model_replica(
        model_uid,
        replica=2,
        model_engine="vllm",
        n_gpu=1,
    )

    result = client.add_model_replica(
        model_uid,
        replica_config={
            "replica_uid": "burst-capacity",
            "devices": [
                {
                    "worker_ip": "192.168.1.12:9978",
                    "n_gpu": 1,
                    "gpu_idx": [1],
                }
            ],
        },
    )

    remaining = client.terminate_model_replica(
        model_uid, replica_id=result["replica_id"]
    )

For multiple replicas, ``replica_config`` may also be a list with one placement
entry per new replica. If any replica in a multi-replica scale-up fails to
launch, Xinference rolls back the replicas created by that operation.

Scale-up is not supported for Xavier-distributed models or models launched with
``n_worker > 1``. Deleting the last replica terminates the running model.

GPU Allocation Strategy
=======================

The current policy is *Idle First*: The scheduler always attempts to assign replicas to the least utilized GPU. Use the ``XINFERENCE_LAUNCH_STRATEGY`` parameter to choose launch strategy.

Set Environment Variables
=========================

.. versionadded:: v1.8.1

Sometimes, we want to specify environment variables for a particular model at runtime.
Since v1.8.1, Xinference provides the capability to configure these individually without needing to set them before starting Xinference.

For Web UI.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/launch_env.png" style="background-color: transparent", width="95%">

When using the command line, use ``--env`` to specify an environment variable.

Example usage:

.. code-block:: bash

  xinference launch xxx --env A 0 --env B 1

Take vLLM as an example: it has versions V1 and V0, and by default, it automatically determines which version to use.
If you want to force the use of V0 by setting ``VLLM_USE_V1=0`` when launching a model, you can specify this during model launching.

Configuring Model Virtual Environment
=====================================

.. versionadded:: v1.8.1

For this part, please refer to :ref:`toggling virtual environments and customizing dependencies <model_launching_virtualenv>`.

Batching / Continuous Batching
==============================

Xinference supports batching for higher throughput. For LLMs on the ``transformers`` engine,
continuous batching is available and can be enabled via environment variables at launch time.

Key settings:

- ``XINFERENCE_BATCH_SIZE`` and ``XINFERENCE_BATCH_INTERVAL`` for general batching behavior.

Example (LLM, transformers):

.. code-block:: bash

  XINFERENCE_BATCH_SIZE=32 XINFERENCE_BATCH_INTERVAL=0.003 xinference-local --log-level debug
  xinference launch -e <endpoint> --model-engine transformers -n qwen1.5-chat -s 4 -f pytorch -q none

For image and video models, batching depends on the selected native engine and model. Diffusers does not use a Xinference step-level scheduler. See :ref:`image` and :ref:`video`.

For detailed behavior, supported models, and aborting requests, see
:ref:`Continuous Batching <user_guide_continuous_batching>`.

Thinking Mode
=============

Some hybrid reasoning models (for example, Qwen3) support an optional *thinking mode*.
You can enable this at launch time via ``--enable-thinking``.

Example usage:

.. code-block:: bash

  xinference launch -n qwen3-xxx --model-engine vllm --enable-thinking

Launch Configuration History
============================

After a model is launched successfully from the Web UI, Xinference stores its
launch configuration in the Supervisor launch-history database and keeps a
user-scoped browser cache for fast access and temporary network failures. When
the deployment dialog is opened again, the newest server-backed configuration
is restored unless the user has already started editing the form.

The configuration history dialog can remove records owned by the current user.
Records used by automatic startup are protected from deletion. In authenticated
deployments, browser caches are separated by the username in the access token;
configuration data owned by another user is not placed in that cache.

The history dialog also lists configurations owned by the current user for
other model names. These records are never applied automatically. After the
user confirms **Use as Template**, Xinference keeps the current model name,
removes the previous model UID and history or automatic-startup metadata, and
copies the remaining launch options into the form without starting a model.

.. _launch_weight_cache_reload:

.. _reconfigure-without-reloading-weights:

Reconfigure and upgrade running models
======================================

.. versionadded:: v4.0.0

.. versionchanged:: 0.4.2

   The default for ``enable_weight_cache`` changed from ``False`` to automatic selection (``None``).

GPU weight retention defaults to automatic selection: supported model environments and configurations enable it; unsupported configurations start normally. Omit ``enable_weight_cache`` or pass ``None`` for automatic selection, set ``True`` to require retention, or ``False`` to disable it. In the deployment dialog, **Keep GPU weights for reload** offers **Auto**, **Enabled** and **Disabled**. Retention requires vLLM 0.31.0+ or SGLang 0.5.21+, Linux with CUDA or ROCm, one worker, one replica, and fixed tensor parallelism. Models with retained weights cannot add replicas.

.. figure:: /_static/keep_weight_for_reload.png
   :alt: Configure GPU weight retention before deploying the model.
   :width: 560px
   :align: center

   Configure GPU weight retention before deploying the model.

.. figure:: /_static/reconfigure_in_running.png
   :alt: Open model reconfiguration from Running Models.
   :width: 100%
   :align: center

   Open model reconfiguration from Running Models.

.. figure:: /_static/reconfigure_in_detail.png
   :alt: Open Reconfigure / upgrade model from the model's detail view.
   :width: 100%
   :align: center

   Open Reconfigure / upgrade model from the model's detail view.

With retained weights enabled, Xinference prepares a replacement model process while the current engine keeps serving. The replacement shares compatible cached weights and builds its own KV cache and CUDA graphs. New requests switch only after loading and a health check succeed; existing requests, including streams, finish on their original engine. Failed preparation leaves the current engine serving. The model UID and GPU reservation stay the same. Terminating the model releases its processes and cached weights.

``POST /v1/models/{model_uid}/reload`` accepts a partial ``model_config`` object and optional ``python_executable`` and ``drain_timeout`` fields. It returns HTTP 202 with an ``operation_id``. Poll ``GET /v1/models/{model_uid}/reload`` until ``status`` is ``ready`` or ``error``. On error, ``restored=true`` means the previous engine is serving. Query ``GET /v1/models/{model_uid}/reload/config`` for supported parameters, current values, engine version and Python executable. Invalid patches return HTTP 400; a reload already in progress returns HTTP 409. ``drain_timeout`` controls the wait for parser-only updates (default 300 seconds, maximum 3600); engine replacements never force existing requests to stop.

Both engines need runtime memory during the switch: shared weights do not share KV cache or CUDA graphs. Reserve enough GPU memory for both engines. Before starting a replacement, Xinference checks free memory while preserving the current engine's configured budget, and rejects insufficient memory or unavailable driver process accounting. Both Xinference environments, and the version that originally launched the model, must support seamless model replacement.

Insufficient memory never triggers an in-place rebuild. Even models launched with the default 0.9 memory fraction need enough spare memory for replacement; otherwise reload fails while inference continues on the current engine.

For consecutive upgrades, ``drain_timeout`` also bounds waiting for previous engine retirement. A timeout leaves current requests running; a later upgrade retries failed retirement.

When authentication is enabled, selecting ``python_executable`` requires administrator access. The current interpreter path is only included in reload metadata for administrators.

To change the model environment, set ``python_executable`` to an installed absolute Python path on the Worker machine. The target must already contain Xinference and the model engine; packages are not installed by reload. Compatible engine and PyTorch versions reuse the retained weights. Different versions use a separate weight cache and need additional GPU memory; upstream compatibility checks remain enabled. In the Web UI, use **Reconfigure / upgrade model** in the running model's detail view.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.reload_model(
        "my-model", {}, python_executable="/opt/xinference-new/bin/python"
    )

.. code-block:: bash

    xinference reload --model-uid my-model \
      --python-executable /opt/xinference-new/bin/python

vLLM supports changing ``max_num_seqs``, ``max_num_batched_tokens``, ``max_model_len``, ``gpu_memory_utilization`` and ``enforce_eager``.

For reasoning-capable models, both engines also accept ``reasoning_content`` through the same reload API. This changes only the Xinference reasoning parser and leaves the engine and GPU weights intact. It works without ``enable_weight_cache`` or the minimum engine versions required for weight retention. Existing requests, including streams, finish before the new setting is applied; new requests wait until the parser change is applied. The ``hot_parameters`` field in reload metadata lists these settings; ``enabled`` still indicates launch-time weight retention. Hot updates also require one worker and one replica. ``weights_reused`` is true only when an engine rebuild uses retained weights.

For vLLM vision models launched with weight retention, ``limit_mm_per_prompt`` can change a positive image count to another positive count, for example ``{"image": 8}``. Unspecified modality limits are preserved in the engine, recovery and automatic-startup configuration. The engine is rebuilt to apply multimodal memory profiling and cache budgets while reusing the retained weights. Setting the image count to zero or enabling images from an initially disabled encoder is rejected; limits for other modalities cannot change through this operation.

SGLang supports changing ``max_running_requests``, ``max_prefill_tokens``, ``context_length``, ``chunked_prefill_size``, ``mem_fraction_static``, ``disable_cuda_graph``, ``cuda_graph_max_bs_decode``, ``cuda_graph_max_bs_prefill`` and ``schedule_conservativeness``.

For SGLang vision models launched with weight retention, ``limit_mm_data_per_request`` accepts a partial image-count patch, for example ``{"image": 8}``. The engine is rebuilt with retained weights so the new tokenizer process applies the limit. Other modality limits are preserved in the engine, recovery and automatic-startup configuration. An initially unlimited or zero image count can become a positive count because this setting only validates request counts; it does not disable the vision encoder. New image limits must be positive integers. Returning to an unlimited or zero image count requires relaunching the model. With an initial zero image count, Xinference skips SGLang's default image warmup.

Parallelism, GPU placement, model path, precision, quantization and weight-layout options cannot change through reload. Pipeline/data parallelism, multi-worker deployments, LoRA, speculative decoding, weight offloading and PD deployments are not supported by this integration yet. Quantized models must pass the upstream IPC compatibility checks. For vLLM, multi-GPU caching uses its native multiprocessing executor. In the running-model detail page, choose **Reconfigure / upgrade model** to edit the supported values and follow progress.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    uid = client.launch_model(
        model_name="qwen2.5-instruct", model_engine="vLLM",
        model_size_in_billions="0_5", model_format="pytorch",
        quantization="none", n_gpu=1, enable_weight_cache=True,
        gpu_memory_utilization=0.4,
    )
    job = client.reload_model(uid, {"max_num_seqs": 32})
    print(job["operation_id"])
    print(client.get_model_reload_status(uid))

.. code-block:: bash

    xinference reload --model-uid MODEL_UID --model-config '{"max_num_seqs": 32}'
    curl http://127.0.0.1:9997/v1/models/MODEL_UID/reload

To change reasoning parsing without restarting the engine:

.. code-block:: python

    job = client.reload_model(uid, {"reasoning_content": True})

For a vision model launched with weight retention and a positive image limit:

.. code-block:: python

    job = client.reload_model(uid, {"limit_mm_per_prompt": {"image": 8}})

To change the image count of a SGLang vision model launched with weight retention:

.. code-block:: python

    job = client.reload_model(uid, {"limit_mm_data_per_request": {"image": 8}})
