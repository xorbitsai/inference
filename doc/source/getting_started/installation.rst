.. _installation:

============
Installation
============
Xinference can be installed with ``pip`` on Linux, Windows, and macOS. To run models using Xinference, you will need to install the backend corresponding to the type of model you intend to serve.

If you aim to serve all supported models, you can install all the necessary dependencies with a single command::

   pip install "xinference[all]"

.. versionchanged:: v1.8.1

   Due to irreconcilable package dependency conflicts between vLLM and sglang, we have removed sglang from the all extra. If you want to use sglang, please install it separately via ``pip install 'xinference[sglang]'``.


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

Notes:

- The transformers engine supports ``pytorch`` / ``gptq`` / ``awq`` / ``bnb`` / ``fp4`` formats.
- FP4 format requires ``transformers`` with ``FPQuantConfig`` support. If you see an import error,
  please upgrade ``transformers`` to a newer version.


vLLM Backend
~~~~~~~~~~~~
vLLM is a fast and easy-to-use library for LLM inference and serving. Xinference will choose vLLM as the backend to achieve better throughput when the following conditions are met:

- The model format is ``pytorch``, ``gptq``, ``awq``, ``fp4``, ``fp8`` or ``bnb``.
- When the model format is ``pytorch``, the quantization is ``none``.
- When the model format is ``awq``, the quantization is ``Int4``.
- When the model format is ``gptq``, the quantization is ``Int3``, ``Int4`` or ``Int8``.
- The system is Linux and has at least one CUDA device
- The model family (for custom models) / model name (for builtin models) is within the list of models supported by vLLM

Currently, supported models include:

.. vllm_start

- ``code-llama``, ``code-llama-instruct``, ``code-llama-python``, ``deepseek``, ``deepseek-chat``, ``deepseek-coder``, ``deepseek-coder-instruct``, ``deepseek-r1-distill-llama``, ``gorilla-openfunctions-v2``, ``HuatuoGPT-o1-LLaMA-3.1``, ``llama-2``, ``llama-2-chat``, ``llama-3``, ``llama-3-instruct``, ``llama-3.1``, ``llama-3.1-instruct``, ``llama-3.3-instruct``, ``tiny-llama``, ``wizardcoder-python-v1.0``, ``wizardmath-v1.0``, ``Yi``, ``Yi-1.5``, ``Yi-1.5-chat``, ``Yi-1.5-chat-16k``, ``Yi-200k``, ``Yi-chat``
- ``codestral-v0.1``, ``mistral-instruct-v0.1``, ``mistral-instruct-v0.2``, ``mistral-instruct-v0.3``, ``mistral-large-instruct``, ``mistral-nemo-instruct``, ``mistral-v0.1``, ``openhermes-2.5``, ``seallm_v2``
- ``Baichuan-M2``, ``codeqwen1.5``, ``codeqwen1.5-chat``, ``deepseek-r1-distill-qwen``, ``DianJin-R1``, ``fin-r1``, ``HuatuoGPT-o1-Qwen2.5``, ``KAT-V1``, ``marco-o1``, ``qwen1.5-chat``, ``qwen2-instruct``, ``qwen2.5``, ``qwen2.5-coder``, ``qwen2.5-coder-instruct``, ``qwen2.5-instruct``, ``qwen2.5-instruct-1m``, ``qwenLong-l1``, ``QwQ-32B``, ``QwQ-32B-Preview``, ``seallms-v3``, ``skywork-or1``, ``skywork-or1-preview``, ``XiYanSQL-QwenCoder-2504``
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
- ``MiniMax-M2``, ``MiniMax-M2.5``
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
Xinference supports models in ``gguf`` format via ``xllamacpp``.
`xllamacpp <https://github.com/xorbitsai/xllamacpp>`_ is developed by Xinference team,
and is the sole backend for llama.cpp since v1.6.0.

.. warning::

    Since Xinference v1.5.0, ``llama-cpp-python`` is deprecated.
    Since Xinference v1.6.0, ``llama-cpp-python`` has been removed.

Initial setup::

   pip install "xinference[llama_cpp]"

For more installation instructions for ``xllamacpp`` to enable GPU acceleration, please refer to: https://github.com/xorbitsai/xllamacpp

SGLang Backend
~~~~~~~~~~~~~~
SGLang has a high-performance inference runtime with RadixAttention. It significantly accelerates the execution of complex LLM programs by automatic KV cache reuse across multiple calls. And it also supports other common techniques like continuous batching and tensor parallelism.

Initial setup::

   pip install "xinference[sglang]"


MLX Backend
~~~~~~~~~~~
MLX-lm is designed for Apple silicon users to run LLM efficiently.

Initial setup::

   pip install "xinference[mlx]"

Other Platforms
~~~~~~~~~~~~~~~

* :ref:`Ascend NPU <installation_npu>`
