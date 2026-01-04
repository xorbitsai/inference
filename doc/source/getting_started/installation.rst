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

