.. _user_guide_pd_separation:

PD separation
=============

Prefill/Decode (PD) separation is available in the community edition. Prefill
replicas compute prompt KV cache; decode replicas reuse it through Xavier and
produce the response. Requests are scheduled round-robin within each role.
No enterprise package or License is required.

Launch
------

Use NVIDIA GPUs and a reachable host address (not ``0.0.0.0``). The V1 connector
requires vLLM 0.21.0 or newer. The existing V0 Xavier adapter is retained for
vLLM versions below 0.11.0; versions 0.11 through 0.20 are not supported by this
V1 connector. GPU integration CI pins vLLM 0.21.0.

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
client, and the CLI ``--replica_config`` JSON option. In the Web UI, select vLLM,
enable per-replica placement, and choose Prefill or Decode for each replica.
The community launch path automatically enables Xavier when P/D roles appear.

CLI launch
----------

Start a local server first, or use an existing supervisor endpoint:

.. code-block:: bash

   xinference-local --host 0.0.0.0 --port 9997

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
* The transferred enterprise implementation provides the Xavier transport;
  ``vllm_transfer_backend_type`` and ``transfer_backend_type`` normalize to
  ``xavier``. They do not enable an additional transport backend.
* The topology is fixed for the deployment lifetime. To change its size or roles,
  terminate the model and relaunch it with the new configuration.

Inference and lifecycle
-----------------------

Send ordinary OpenAI-compatible chat/completion requests to the base model UID.
Streaming and non-streaming requests use the same PD route. The prefill subrequest
uses one output token without changing the decode request's generation settings.
Routing becomes available only after all replicas and transfer components are
ready. Aborting a request reaches both roles. Terminating a deployment also
removes its router, rank-zero coordinator, collective manager, and block tracker.

Verification
------------

On a machine with two free NVIDIA GPUs and the pinned vLLM environment, run:

.. code-block:: bash

   XINFERENCE_TEST_PD_GPU=1 python -m pytest -v \
     xinference/model/llm/vllm/xavier/test/test_pd_gpu.py

This test launches a community deployment in real subprocesses with one GPU per
role. It checks streaming and non-streaming responses, repeated prompts, and four
concurrent requests, requires both producer staging and decoder KV-load log
evidence, and terminates the deployment. To use locally cached weights, set
``XINFERENCE_TEST_PD_MODEL_PATH``, ``XINFERENCE_TEST_PD_MODEL_NAME``, and
``XINFERENCE_TEST_PD_MODEL_SIZE`` to the model path and its registered model name
and size. The manually triggered ``PD GPU integration`` GitHub
Actions workflow runs the same test on a selected runner with two GPUs.

Hybrid/recurrent attention limitation
-------------------------------------

Xavier PD currently rejects hybrid/recurrent attention caches, including Qwen3.5.
The transferred implementation does not yet reliably preserve their recurrent
state across prefix-cache reuse and concurrent requests. Use ordinary single
instances for these models, or a full-attention model such as Qwen3 for PD.
Successful launch alone is not evidence of correct hybrid-state transfer.

V1 supported configurations and cache safety
-------------------------------------------

The V1 connector currently requires one GPU per replica (TP=1, PP=1), text-only
models, and no LoRA adapters. Multimodal models, prompt embeddings and salted
prompts are rejected until their cache identities and partitioning are supported.

CPU snapshots are keyed by prompt content rather than reusable GPU block IDs.
Readers reserve complete snapshots during transfer; cache pressure or a missing
snapshot falls back to local computation. The number of retained CPU blocks is
bounded by the engine's KV block capacity. CPU memory can approach the GPU KV
cache size (twice that size for BF16 transported as FP32), in addition to in-flight
transfer buffers. Only complete layers are published for remote reuse.

The connector handles block-first and K/V-first attention cache layouts. A
layout whose block axis cannot be identified safely is rejected. V0 prefill
replicas release only the requested completed sequence; ordinary hybrid and
decode replicas retain normal automatic cleanup.
