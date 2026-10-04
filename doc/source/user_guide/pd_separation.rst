.. _user_guide_pd_separation:

PD separation
=============

Prefill/Decode (PD) separation is available in the community edition. Prefill
replicas compute prompt KV cache; decode replicas reuse it through Xavier or native NIXL and
produce the response. Requests are scheduled round-robin within each role.
No enterprise package or License is required.

Launch
------

Xavier requires NVIDIA GPUs, a reachable host address (not ``0.0.0.0``), and vLLM 0.21.0 or newer. Older deployments can use the legacy Xavier integration with vLLM below 0.11.0; versions 0.11 through 0.20 are not supported.

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
The community launch path defaults to Xavier when P/D roles appear.

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

Xavier PD does not support hybrid/recurrent attention models such as Qwen3.5. Deploy these models without PD, or choose a full-attention model such as Qwen3 for PD.

Xavier requirements and memory
------------------------------

Xavier V1 requires one GPU per replica (TP=1, PP=1) and text-only models without LoRA. Multimodal models, prompt embeddings and salted prompts are not supported.

Xavier's CPU cache can consume as much memory as the engine's GPU KV cache. Allow additional host memory for transfers when sizing your deployment.

GPU-first Xavier cache (experimental)
-------------------------------------

GPU-first caching requires the Xavier backend and more than one replica.

For the Xavier backend, set ``xavier_gpu_cache_bytes`` to a non-negative integer, such as ``268435456`` for 256 MiB per replica. Omit it to use the existing CPU cache, or set it to ``0`` to use GPU-first caching with no GPU storage budget. This option does not select the native vLLM NIXL backend.

Install ``xoscar[nixl]>=0.11.1`` in both the worker and model environments. GPU-first caching requires Linux, NVIDIA CUDA, vLLM 0.21.0 or newer, a full-attention model and one GPU per replica (TP=1, PP=1). Use the same model, cache dtype and engine configuration on all replicas.

The budget applies only to Xavier's GPU cache. Leave additional GPU memory for the model, vLLM's own KV cache and transfers. When this cache fills, Xavier stores reusable data in CPU memory; if no cache space is available, new data may not be retained for reuse.

With GPU-first caching, cache transfers can overlap with response generation for other requests.

Transfer failures and incompatible KV cache layouts raise errors instead of silently recomputing the request.

For implementation details, profiling and contributor tests, see the `Xavier developer README <https://github.com/xorbitsai/inference/blob/main/xinference/model/llm/vllm/xavier/README.md>`_.
