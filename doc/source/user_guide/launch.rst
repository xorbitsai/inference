.. _launch:

============================
Model Launching Instructions
============================

This document aims to provide a functional overview of model launching.

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

Estimates use offline metadata, one sequence, 2048 tokens, FP16 KV cache and
80% of currently free memory. Automatic GPU placement uses the lowest free memory
among eligible GPUs; CPU and Apple Silicon use available system memory. Multiple
GPUs are not added together: multi-GPU requests keep compatibility-only selection.
Known oversized candidates are excluded. Missing data remains unverified. Longer
contexts, concurrency and engine preallocation can need more memory. No resources
are reserved, and you still review and launch the model yourself.

Using the API
-------------

``POST /v1/models/recommend`` is read-only. Supply ``model_name`` and
``model_type="LLM"``; the optional ``constraints`` object accepts
``model_size_in_billions``, ``worker_ip``, ``enable_virtual_env``, ``n_gpu``, and
``gpu_idx``. Supplied constraints are hard requirements, never silently relaxed.

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

Replica
=======

During deployment, the progress endpoint ``GET /v1/models/{model_uid}/progress``
reports a stage for each replica. After model files finish downloading, a replica
can report ``waiting_for_dependencies`` while another launch prepares its shared
virtual environment, then ``installing_dependencies`` while packages are being
installed, and ``loading`` after the environment is ready. The overall
percentage returned by the endpoint covers the whole launch. The Web UI shows
progress for each replica without an overall launch bar. Dependency installation
has no separate package-level percentage.

The Web UI shows each replica's file download details inside its replica card.

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
