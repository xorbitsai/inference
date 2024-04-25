.. _installation:

============
Installation
============
Xinference can be installed with ``pip`` on Linux, Windows, and macOS. To run models using Xinference, you will need to install the backend corresponding to the type of model you intend to serve.

If you aim to serve all supported models, you can install all the necessary dependencies with a single command::

   pip install "xinference[all]"

.. note::
   If you want to serve models in GGML format, it's advised to install the GGML dependencies manually based on your hardware specifications to enable acceleration. For more details, see the :ref:`installation_ggml` section.


If you want to install only the necessary backends, here's a breakdown of how to do it.

Transformers Backend
~~~~~~~~~~~~~~~~~~~~
PyTorch (transformers) supports the inference of most state-of-art models. It is the default backend for models in PyTorch format::

   pip install "xinference[transformers]"


vLLM Backend
~~~~~~~~~~~~
vLLM is a fast and easy-to-use library for LLM inference and serving. Xinference will choose vLLM as the backend to achieve better throughput when the following conditions are met:

- The model format is ``pytorch``, ``gptq`` or ``awq``.
- When the model format is ``pytorch``, the quantization is ``none``.
- When the model format is ``awq``, the quantization is ``Int4``.
- When the model format is ``gptq``, the quantization is ``Int3``, ``Int4`` or ``Int8``.
- The system is Linux and has at least one CUDA device
- The model family (for custom models) / model name (for builtin models) is within the list of models supported by vLLM

Currently, supported models include:

.. vllm_start

- ``llama-2``, ``llama-3``, ``llama-2-chat``, ``llama-3-instruct``
- ``baichuan``, ``baichuan-chat``, ``baichuan-2-chat``
- ``internlm-16k``, ``internlm-chat-7b``, ``internlm-chat-8k``, ``internlm-chat-20b``
- ``mistral-v0.1``, ``mistral-instruct-v0.1``, ``mistral-instruct-v0.2``
- ``Yi``, ``Yi-chat``
- ``code-llama``, ``code-llama-python``, ``code-llama-instruct``
- ``c4ai-command-r-v01``, ``c4ai-command-r-v01-4bit``
- ``vicuna-v1.3``, ``vicuna-v1.5``
- ``internlm2-chat``
- ``qwen-chat``
- ``mixtral-instruct-v0.1``
- ``chatglm3``, ``chatglm3-32k``, ``chatglm3-128k``
- ``deepseek-chat``, ``deepseek-coder-instruct``
- ``qwen1.5-chat``, ``qwen1.5-moe-chat``
- ``codeqwen1.5-chat``
- ``gemma-it``
- ``orion-chat``, ``orion-chat-rag``
.. vllm_end

To install Xinference and vLLM::

   pip install "xinference[vllm]"

.. _installation_ggml:

GGML Backend
~~~~~~~~~~~~
It's advised to install the GGML dependencies manually based on your hardware specifications to enable acceleration.

Initial setup::

   pip install xinference

Hardware-Specific installations:

- Apple Silicon::

   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

- Nvidia cards::

   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

- AMD cards::

   CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
