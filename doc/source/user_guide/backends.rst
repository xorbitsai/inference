.. _user_guide_backends:

========
Backends
========

Xinference supports multiple backends for different models. After the user specifies the model,
xinference will automatically select the appropriate backend.

llama.cpp
~~~~~~~~~
`llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_ is the python binding of
`llama.cpp`. `llama-cpp` is developed based on the tensor library `ggml`, supporting inference of
the LLaMA series models and their variants.

We recommend that users install `llama-cpp-python` on the worker themselves and adjust the `cmake`
parameters according to the hardware to achieve the best inference efficiency. Please refer to the
`llama-cpp-python installation guide <https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal>`_.


transformers
~~~~~~~~~~~~
Transformers supports the inference of most state-of-art models. It is the default backend for models in PyTorch format.

vLLM
~~~~
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

- ``llama-2``, ``llama-3``, ``llama-2-chat``, ``llama-3-instruct``
- ``baichuan``, ``baichuan-chat``, ``baichuan-2-chat``
- ``internlm-16k``, ``internlm-chat-7b``, ``internlm-chat-8k``, ``internlm-chat-20b``
- ``mistral-v0.1``, ``mistral-instruct-v0.1``, ``mistral-instruct-v0.2``
- ``Yi``, ``Yi-1.5``, ``Yi-chat``, ``Yi-1.5-chat``
- ``code-llama``, ``code-llama-python``, ``code-llama-instruct``
- ``deepseek``, ``deepseek-coder``, ``deepseek-chat``, ``deepseek-coder-instruct``
- ``codeqwen1.5``, ``codeqwen1.5-chat``
- ``c4ai-command-r-v01``, ``c4ai-command-r-v01-4bit``
- ``vicuna-v1.3``, ``vicuna-v1.5``
- ``internlm2-chat``
- ``qwen-chat``
- ``mixtral-instruct-v0.1``, ``mixtral-8x22B-instruct-v0.1``
- ``chatglm3``, ``chatglm3-32k``, ``chatglm3-128k``
- ``qwen1.5-chat``, ``qwen1.5-moe-chat``
- ``gemma-it``
- ``orion-chat``, ``orion-chat-rag``
.. vllm_end

SGLang
~~~~~~
`SGLang <https://github.com/sgl-project/sglang>`_ has a high-performance inference runtime with RadixAttention.
It significantly accelerates the execution of complex LLM programs by automatic KV cache reuse across multiple calls.
And it also supports other common techniques like continuous batching and tensor parallelism.


