.. _installation:

============
Installation
============
Xinference can be installed with ``pip`` on Linux, Windows, and macOS. To run models using Xinference, you will need to install the backend corresponding to the type of model you intend to serve.

If you aim to serve all supported models, you can install all the necessary dependencies with a single command::

   pip install "xinference[all]"


Several usage scenarios require special attention.

.. admonition:: **GGUF format** with **llama.cpp engine**

   In this situation, it's advised to install its dependencies manually based on your hardware specifications to enable acceleration. For more details, see the :ref:`installation_gguf` section.

.. admonition:: **AWQ or GPTQ** format with **transformers engine**

   **This section is added in v1.6.0.**

   This is because the dependencies at this stage require special options and are difficult to install. Please run command below in advance

   .. code-block:: bash

      pip install "xinference[transformers_quantization]" --no-build-isolation

   Some dependencies like ``transformers`` might be downgraded, you can run ``pip install "xinference[all]"`` afterwards.


If you want to install only the necessary backends, here's a breakdown of how to do it.

.. _inference_backend:

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

- ``llama-2``, ``llama-3``, ``llama-3.1``, ``llama-3.2-vision``, ``llama-2-chat``, ``llama-3-instruct``, ``llama-3.1-instruct``, ``llama-3.3-instruct``
- ``mistral-v0.1``, ``mistral-instruct-v0.1``, ``mistral-instruct-v0.2``, ``mistral-instruct-v0.3``, ``mistral-nemo-instruct``, ``mistral-large-instruct``
- ``codestral-v0.1``
- ``Yi``, ``Yi-1.5``, ``Yi-chat``, ``Yi-1.5-chat``, ``Yi-1.5-chat-16k``
- ``code-llama``, ``code-llama-python``, ``code-llama-instruct``
- ``deepseek``, ``deepseek-coder``, ``deepseek-chat``, ``deepseek-coder-instruct``, ``deepseek-r1-distill-qwen``, ``deepseek-v2-chat``, ``deepseek-v2-chat-0628``, ``deepseek-v2.5``, ``deepseek-v3``, ``deepseek-r1``, ``deepseek-r1-distill-llama``
- ``yi-coder``, ``yi-coder-chat``
- ``codeqwen1.5``, ``codeqwen1.5-chat``
- ``qwen2.5``, ``qwen2.5-coder``, ``qwen2.5-instruct``, ``qwen2.5-coder-instruct``, ``qwen2.5-instruct-1m``
- ``baichuan-2-chat``
- ``internlm2-chat``
- ``internlm2.5-chat``, ``internlm2.5-chat-1m``
- ``qwen-chat``
- ``mixtral-instruct-v0.1``, ``mixtral-8x22B-instruct-v0.1``
- ``chatglm3``, ``chatglm3-32k``, ``chatglm3-128k``
- ``glm4-chat``, ``glm4-chat-1m``, ``glm4-0414``
- ``codegeex4``
- ``qwen1.5-chat``, ``qwen1.5-moe-chat``
- ``qwen2-instruct``, ``qwen2-moe-instruct``
- ``QwQ-32B-Preview``, ``QwQ-32B``
- ``marco-o1``
- ``fin-r1``
- ``seallms-v3``
- ``skywork-or1-preview``
- ``gemma-it``, ``gemma-2-it``, ``gemma-3-1b-it``
- ``orion-chat``, ``orion-chat-rag``
- ``c4ai-command-r-v01``
- ``minicpm3-4b``
- ``internlm3-instruct``
- ``moonlight-16b-a3b-instruct``
- ``qwen3``
.. vllm_end

To install Xinference and vLLM::

   pip install "xinference[vllm]"
   
   # FlashInfer is optional but required for specific functionalities such as sliding window attention with Gemma 2.
   # For CUDA 12.4 & torch 2.4 to support sliding window attention for gemma 2 and llama 3.1 style rope
   pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
   # For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html
   

.. _installation_gguf:

Llama.cpp Backend
~~~~~~~~~~~~~~~~~
Xinference supports models in ``gguf`` format via ``xllamacpp`` or ``llama-cpp-python``.
`xllamacpp <https://github.com/xorbitsai/xllamacpp>`_ is developed by Xinference team,
and will be the sole backend for llama.cpp in the future.

.. note::

    ``xllamacpp`` is the default option for llama.cpp backend since v1.5.0.
    To enable ``llama-cpp-python``, add environment variable ``USE_XLLAMACPP=0``.

.. warning::

    Since Xinference v1.5.0, ``llama-cpp-python`` will be deprecated.
    For Xinference v1.6.0, ``llama-cpp-python`` will be removed.

Initial setup::

   pip install xinference

Installation instructions for ``xllamacpp``:

- CPU or Mac Metal::

   pip install -U xllamacpp

- CUDA::

   pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124

- HIP::

   pip install xllamacpp --force-reinstall --index-url https://xorbitsai.github.io/xllamacpp/whl/rocm-6.0.2

Hardware-Specific installations for ``llama-cpp-python``:

- Apple Silicon::

   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

- Nvidia cards::

   CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

- AMD cards::

   CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python


SGLang Backend
~~~~~~~~~~~~~~
SGLang has a high-performance inference runtime with RadixAttention. It significantly accelerates the execution of complex LLM programs by automatic KV cache reuse across multiple calls. And it also supports other common techniques like continuous batching and tensor parallelism.

Initial setup::

   pip install "xinference[sglang]"

   # For CUDA 12.4 & torch 2.4 to support sliding window attention for gemma 2 and llama 3.1 style rope
   pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
   # For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html


MLX Backend
~~~~~~~~~~~
MLX-lm is designed for Apple silicon users to run LLM efficiently.

Initial setup::

   pip install "xinference[mlx]"

Other Platforms
~~~~~~~~~~~~~~~

* :ref:`Ascend NPU <installation_npu>`

