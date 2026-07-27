.. _user_guide_backends:

========
Backends
========

Xinference supports multiple backends for different models. After the user specifies the model,
xinference will automatically select the appropriate backend.

.. _llama_cpp_backend:

llama.cpp
=========

Xinference now supports `xllamacpp <https://github.com/xorbitsai/xllamacpp>`_ which developed by Xinference team
to run llama.cpp backend.
`llama.cpp` is developed based on the tensor library `ggml`, supporting inference of
the LLaMA series models and their variants.

.. warning::

    Since Xinference v1.5.0,
    ``xllamacpp`` becomes default option for llama.cpp, and ``llama-cpp-python`` is deprecated.
    Since Xinference v1.6.0, ``llama-cpp-python`` has been removed.

Automatic GPU wheel selection
-----------------------------

.. versionadded:: 3.0

When per-model virtual environments are enabled and a usable NVIDIA GPU is
present, Xinference automatically installs the ``xllamacpp`` wheel matching
the detected CUDA line. CUDA 12.8 and later in the 12.x line use the ``cu128``
wheel index, while CUDA 13.x uses the ``cu132`` wheel index. CPU-only hosts and
unsupported CUDA versions keep the default CPU wheel from PyPI. No separate
``xllamacpp`` GPU installation command is required for this launch path.


For all configurable llama.cpp parameters, please refer to the definition of the ``common_params`` structure in ``llama.cpp`` `common.h <https://github.com/ggml-org/llama.cpp/blob/master/common/common.h>`_

There may be some nested parameters. For example, ``sampling.top_k``. Just use the ``.`` to separate nested parameters.

Here is an example of setting nested sampling parameters in WebUI:

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/xllamacpp_param.png" style="background-color: transparent", width="95%">

Auto NGL
-------------

.. versionadded:: v1.6.1
    Auto GPU layers estimation is enabled since v1.6.1 when ``n-gpu-layers`` is not specified (default is -1).

This feature automatically detects the number of GPU layers (NGL) for the llama.cpp backend. Please be aware that this
is not an accurate calculation. Therefore, the ``-ngl`` result might not be the most optimized, and there is still a
chance of encountering an out-of-memory error.

Currently, there is no official implementation for auto ngl. Please refer to the following issues for more information:

- https://github.com/ggml-org/llama.cpp/issues/13860
- https://github.com/ggml-org/llama.cpp/pull/6502

Our implementation is based on the Ollama auto ngl, but there are some differences:

- We utilize device information detected by `xllamacpp <https://github.com/xorbitsai/xllamacpp>`_.
- We have removed support for less popular architectures, these architectures will use the default calculation.
- We fall back to offloading all the layers to the GPU if the auto ngl fails.
- We do not support multimodal projectors embedded into the model GGUF, as this is a very experimental feature.


Common Issues
-------------

- **Server error: {'code': 500, 'message': 'failed to process image', 'type': 'server_error'}**

  The error logs from server:

  .. code-block::

    encoding image or slice...
    slot update_slots: id  0 | task 0 | kv cache rm [10, end)
    srv  process_chun: processing image...
    ggml_metal_graph_compute: command buffer 0 failed with status 5
    error: Internal Error (0000000e:Internal Error)
    clip_image_batch_encode: ggml_backend_sched_graph_compute failed with error -1
    failed to encode image
    srv  process_chun: image processed in 2288 ms
    mtmd_helper_eval failed with status 1
    slot update_slots: id  0 | task 0 | failed to process image, res = 1

  This could be caused by running out of memory. You can try reducing memory usage by decreasing ``n_ctx``.

- **Server error: {'code': 400, 'message': 'the request exceeds the available context size. try increasing the context size or enable context shift', 'type': 'invalid_request_error'}**

  If you are using the multimodal feature, the ``ctx_shift`` is disabled by default. Please increase the context size by
  either increasing ``n_ctx`` or reducing ``n_parallel``.

- **Server error: {'code': 500, 'message': 'Input prompt is too big compared to KV size. Please try increasing KV size.', 'type': 'server_error'}**

  The error logs from server:

  .. code-block::

    ggml_metal_graph_compute: command buffer 1 failed with status 5
    error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
    graph_compute: ggml_backend_sched_graph_compute_async failed with error -1
    llama_decode: failed to decode, ret = -3
    srv  update_slots: failed to decode the batch: KV cache is full - try increasing it via the context size, i = 0, n_batch = 2048, ret = -3

  This could be caused by the KV cache allocation failure. You can try to reduce the context size by either reducing
  ``n_ctx`` or increasing ``n_parallel``, or loading a partial model onto the GPU by adjusting ``n_gpu_layers``. Be aware
  that if you are handling inference requests serially, increasing ``n_parallel`` can't improve the latency or throughput.

transformers
============
Transformers supports the inference of most state-of-art models. It is the default backend for models in PyTorch format.

.. _vllm_backend:

vLLM
====
vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with PagedAttention
- Continuous batching of incoming requests
- Optimized CUDA kernels

When the following conditions are met, Xinference will choose vLLM as the inference engine:

- The model format is ``pytorch``, ``gptq``, ``awq``, ``fp4``, ``fp8`` or ``bnb``.
- When the model format is ``pytorch``, the quantization is ``none``.
- When the model format is ``awq``, the quantization is ``Int4``.
- When the model format is ``gptq``, the quantization is ``Int3``, ``Int4`` or ``Int8``.
- The system is Linux and has at least one CUDA device
- The model family (for custom models) / model name (for builtin models) is within the list of models supported by vLLM

Currently, supported model includes:

.. vllm_start

- ``code-llama``, ``code-llama-instruct``, ``code-llama-python``, ``deepseek``, ``deepseek-chat``, ``deepseek-coder``, ``deepseek-coder-instruct``, ``deepseek-r1-distill-llama``, ``gorilla-openfunctions-v2``, ``HuatuoGPT-o1-LLaMA-3.1``, ``llama-2``, ``llama-2-chat``, ``llama-3``, ``llama-3-instruct``, ``llama-3.1``, ``llama-3.1-instruct``, ``llama-3.3-instruct``, ``minicpm5-1b``, ``tiny-llama``, ``wizardcoder-python-v1.0``, ``wizardmath-v1.0``, ``Yi``, ``Yi-1.5``, ``Yi-1.5-chat``, ``Yi-1.5-chat-16k``, ``Yi-200k``, ``Yi-chat``
- ``codestral-v0.1``, ``mistral-instruct-v0.1``, ``mistral-instruct-v0.2``, ``mistral-instruct-v0.3``, ``mistral-large-instruct``, ``mistral-nemo-instruct``, ``mistral-v0.1``, ``openhermes-2.5``, ``seallm_v2``
- ``Baichuan-M2``, ``codeqwen1.5``, ``codeqwen1.5-chat``, ``deepseek-r1-distill-qwen``, ``DianJin-R1``, ``fin-r1``, ``HuatuoGPT-o1-Qwen2.5``, ``KAT-V1``, ``marco-o1``, ``qwen1.5-chat``, ``qwen2-instruct``, ``qwen2.5``, ``qwen2.5-coder``, ``qwen2.5-coder-instruct``, ``qwen2.5-instruct``, ``qwen2.5-instruct-1m``, ``qwenLong-l1``, ``QwQ-32B``, ``QwQ-32B-Preview``, ``seallms-v3``, ``skywork-or1``, ``skywork-or1-preview``, ``vibethinker``, ``XiYanSQL-QwenCoder-2504``
- ``llama-3.2-vision``, ``llama-3.2-vision-instruct``
- ``baichuan-2``, ``baichuan-2-chat``
- ``InternLM2ForCausalLM``
- ``qwen-chat``
- ``mixtral-8x22B-instruct-v0.1``, ``mixtral-instruct-v0.1``, ``mixtral-v0.1``
- ``cogagent``
- ``glm-edge-chat``, ``glm4-chat``, ``glm4-chat-1m``
- ``codegeex4``, ``glm-4v``
- ``seallm_v2.5``
- ``orion-chat``
- ``qwen1.5-moe-chat``, ``qwen2-moe-instruct``
- ``CohereForCausalLM``
- ``deepseek-v2-chat``, ``deepseek-v2-chat-0628``, ``deepseek-v2.5``, ``deepseek-vl2``
- ``deepseek-prover-v2``, ``deepseek-r1``, ``deepseek-r1-0528``, ``deepseek-v3``, ``deepseek-v3-0324``, ``Deepseek-V3.1``, ``moonlight-16b-a3b-instruct``
- ``deepseek-r1-0528-qwen3``, ``qwen3``
- ``minicpm3-4b``
- ``internlm3-instruct``
- ``gemma-3-1b-it``
- ``glm4-0414``
- ``minicpm-2b-dpo-bf16``, ``minicpm-2b-dpo-fp16``, ``minicpm-2b-dpo-fp32``, ``minicpm-2b-sft-bf16``, ``minicpm-2b-sft-fp32``, ``minicpm4``
- ``Ernie4.5``
- ``Qwen3-Coder``, ``Qwen3-Instruct``, ``Qwen3-Thinking``
- ``glm-4.5``, ``GLM-4.6``, ``GLM-4.7``
- ``gpt-oss``
- ``seed-oss``
- ``Qwen3-Next-Instruct``, ``Qwen3-Next-Thinking``
- ``DeepSeek-V3.2``, ``DeepSeek-V3.2-Exp``
- ``MiniMax-M2``, ``MiniMax-M2.5``, ``MiniMax-M2.7``
- ``GLM-4.7-Flash``
- ``glm-5``, ``glm-5.1``
- ``DeepSeek-V4-Flash``, ``DeepSeek-V4-Pro``
.. vllm_end

Besides LLMs, vLLM can also serve embedding models. Model families whose
names start with ``bge``, ``gte``, ``text2vec``, ``m3e``, ``Qwen3``, or
``bce`` can be launched with ``--model-engine vllm`` — see
:ref:`Embeddings <embed>`.

.. _sglang_backend:

SGLang
======
`SGLang <https://github.com/sgl-project/sglang>`_ has a high-performance inference runtime with RadixAttention.
It significantly accelerates the execution of complex LLM programs by automatic KV cache reuse across multiple calls.
And it also supports other common techniques like continuous batching and tensor parallelism.

.. _mlx_backend:

MLX
===
`MLX <https://github.com/ml-explore/mlx-examples/tree/main/llms>`_ provides efficient runtime
to run LLM on Apple silicon. It's recommended to use for Mac users when running on Apple silicon
if the model has MLX format support.


.. _speculative_decoding:

Speculative decoding
====================
Some models ship a small paired drafter checkpoint that predicts several tokens
ahead, which the target model then verifies in one pass. Output is unchanged,
decoding gets faster. Gemma 4 calls this multi-token prediction (MTP) and
publishes a ``*-it-assistant`` drafter for every variant.

Pass ``--enable_mtp true`` at launch to download the drafter declared by the
model spec and run it alongside the target model:

.. code-block:: bash

   xinference launch --model-name gemma-4 --model-engine vllm \
     --model-format pytorch --size-in-billions 12 --quantization none \
     --enable_mtp true

In the Web UI the same options live under *Advanced Configuration → Speculative
Decoding*, which only appears for a format/size that actually ships a drafter.

Optional parameters:

* ``--num_speculative_tokens <n>``: how many tokens the drafter proposes per
  round, including the bonus token. Left unset, three things can happen. MLX
  reads it from the drafter, which runs at the depth it was trained for (``4``
  for Gemma 4). llama.cpp keeps xllamacpp's own default (``3`` as of
  ``2026.7``), since it has one. vLLM and SGLang follow the Gemma 4 recipe by
  model size: ``2`` for E2B, ``4`` for E4B and 26B-A4B, and the lower end
  (``4``) of the recommended ``4-8`` range for 12B and 31B.
* ``--draft_quantization <quantization>``: which drafter conversion to use, when
  the spec declares more than one — the MLX build of Gemma 4 12B publishes
  eight. Defaults to the first declared, which is the least quantized one: a
  drafter is small and quantizing it costs acceptance rate, so pairing a
  quantized target with an unquantized drafter is the recommended setup.
* ``--draft_model_path <path>``: use a local drafter instead of the one declared
  by the spec. Implies ``--enable_mtp true``.

The drafter must match the target model family and size — it shares the target's
KV cache, so a mismatched checkpoint is rejected rather than silently degrading.

Engine support:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Engine
     - Requirement
     - Notes
   * - :ref:`vLLM <vllm_backend>`
     - ``vllm>=0.22.0``, ``transformers>=5.8.0``
     - Translated into ``speculative_config`` with ``method: mtp``. An
       explicitly provided ``speculative_config`` is left untouched. Older
       vLLM treats the drafter as a generic draft model, while older
       Transformers does not recognize ``gemma4_assistant``; either case is
       rejected before engine initialization. Virtual-environment launches
       also synchronize ``flashinfer-cubin`` with ``flashinfer-python`` before
       starting vLLM, repairing stale environments that contain mismatched
       FlashInfer packages.
   * - :ref:`SGLang <sglang_backend>`
     - ``sglang==0.5.13.post1``, ``transformers==5.8.1``
     - Translated into ``--speculative-algorithm NEXTN`` with the matching
       ``speculative_num_steps`` / ``speculative_num_draft_tokens``. An
       explicitly provided ``speculative_algorithm`` is left untouched.
   * - :ref:`MLX <mlx_backend>`
     - ``mlx-vlm>=0.5.0`` (``>=0.6.1`` for Gemma 4 12B)
     - Served by the MLX vision engine, the one that runs multimodal models
       such as Gemma 4. The drafter is validated against the target when the
       model loads.
   * - :ref:`llama.cpp <llama_cpp_backend>`
     - ``xllamacpp>=2026.6.9713``
     - Translated into the ``draft-mtp`` speculative implementation. The
       drafter is a single gguf published inside the target's own repository,
       so its quantizations are its own (``BF16``, ``F16``, ``Q8_0`` for Gemma
       4) and independent of the target's. Earlier llama.cpp builds do not
       know the ``gemma4-assistant`` architecture and cannot load it.

Not supported by the Transformers engine: it runs its own continuous-batching
loop rather than ``generate()``, so there is nowhere to attach a drafter.

When it pays off
----------------
Speculation is only worth it when a drafting step is cheap *relative to* a
target decoding step. The drafter is a small dense model whose cost does not
change with the target, so the ratio is what decides the outcome — and a
mixture-of-experts target can land on the wrong side of it, because only its
activated slice is read per token:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - gemma-4 MLX, 4bit, M5 Pro
     - Without a drafter
     - With MTP
     - Accepted per round
   * - 31B (dense, 18.4 GB read per token)
     - 14.9 tok/s
     - **31.0 tok/s** (2.1x)
     - 2.08 of 4
   * - 26B-A4B (MoE, ~2.2 GB read per token)
     - 73.2 tok/s
     - 65.9 tok/s (0.9x)
     - 1.40 of 4

The 0.83 GB drafter costs about 5% of a 31B decoding step but 39% of a
26B-A4B one, so on the MoE the three drafting steps of a round already exceed
one plain decoding step — and the round has to win that back from a lower
acceptance rate. The MoE is still the faster model here in absolute terms; it
simply has no headroom left for speculation.

So measure before leaving it on, and note that the verification step itself is
not the problem: a four-token forward costs only ~40% more than a single-token
one on either model.
