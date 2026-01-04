.. _user_guide_backends:

========
Backends
========

Xinference supports multiple backends for different models. After the user specifies the model,
xinference will automatically select the appropriate backend.

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

- The model format is ``pytorch``, ``gptq`` or ``awq``.
- When the model format is ``pytorch``, the quantization is ``none``.
- When the model format is ``awq``, the quantization is ``Int4``.
- When the model format is ``gptq``, the quantization is ``Int3``, ``Int4`` or ``Int8``.
- The system is Linux and has at least one CUDA device
- The model family (for custom models) / model name (for builtin models) is within the list of models supported by vLLM

Currently, supported model includes:

.. vllm_start

- ``LlamaForCausalLM``, ``LlamaForCausalLM``
- ``MistralForCausalLM``, ``MistralForCausalLM``
- ``Qwen2ForCausalLM``, ``Qwen2ForCausalLM``
- ``MllamaForConditionalGeneration``
- ``BaichuanForCausalLM``
- ``InternLM2ForCausalLM``
- ``QWenLMHeadModel``
- ``MixtralForCausalLM``
- ``ChatGLMForConditionalGeneration``
- ``GlmForCausalLM``
- ``ChatGLMModel``
- ``GemmaForCausalLM``
- ``OrionForCausalLM``
- ``Qwen2MoeForCausalLM``
- ``CohereForCausalLM``
- ``DeepseekV2ForCausalLM``
- ``DeepseekV3ForCausalLM``
- ``Qwen3ForCausalLM``
- ``MiniCPM3ForCausalLM``
- ``InternLM3ForCausalLM``
- ``Gemma3ForCausalLM``
- ``Glm4ForCausalLM``
- ``MiniCPMForCausalLM``
- ``Ernie4_5ForCausalLM``
- ``Qwen3MoeForCausalLM``
- ``Glm4MoeForCausalLM``
- ``GptOssForCausalLM``
- ``SeedOssForCausalLM``
- ``Qwen3NextForCausalLM``
- ``DeepseekV32ForCausalLM``
- ``MiniMaxM2ForCausalLM``
.. vllm_end

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


