.. _models_llm_deepseek-v4-flash-0731:

========================================
DeepSeek-V4-Flash-0731
========================================

- **Context Length:** 1048576
- **Model Name:** DeepSeek-V4-Flash-0731
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** Official DeepSeek-V4-Flash release with enhanced agentic capabilities and an attached DSpark speculative decoding module.

This is a separate built-in model entry from the existing ``DeepSeek-V4-Flash``
preview entry. It represents the native FP8 checkpoint and is intended to run
with vLLM 0.20.1 or newer.

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (fp8, 304 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 304
- **Quantizations:** fp8
- **Engines:** vLLM (vLLM >= 0.20.1)
- **Architecture:** ``DeepseekV4ForCausalLM``
- **Model ID:** deepseek-ai/DeepSeek-V4-Flash-0731
- **Model Hubs:** `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731>`__

Launch the model with the native FP8 format:

.. code-block:: bash

   xinference launch \
       --model-engine vLLM \
       --model-name DeepSeek-V4-Flash-0731 \
       --size-in-billions 304 \
       --model-format fp8 \
       --quantization fp8

The model repository supplies DeepSeek-V4-specific encoding code. For safety,
Xinference only loads this repository code when the operator enables:

.. code-block:: bash

   export XINFERENCE_TRUST_REMOTE_CODE=1

Reasoning and thinking mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^

DeepSeek-V4 uses its repository ``encoding/encoding_dsv4.py`` implementation.
The ``chat_template_kwargs`` option can select thinking mode and reasoning
level, for example:

.. code-block:: json

   {
     "chat_template_kwargs": {
       "enable_thinking": true,
       "reasoning_effort": "high"
     }
   }

Set ``enable_thinking`` to ``false`` to request chat mode. Supported reasoning
levels depend on the model repository encoding implementation; ``low``,
``high`` and ``max`` are forwarded without Xinference rewriting them.

DSpark speculative decoding (opt-in)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The checkpoint includes a DSpark speculative decoding module, but Xinference
does not enable it automatically. Enable it explicitly through vLLM model
configuration when the deployment hardware and installed vLLM version support
it:

.. code-block:: json

   {
     "speculative_config": {
       "method": "dspark",
       "num_speculative_tokens": 7,
       "draft_sample_method": "greedy"
     }
   }

The following parameters are an upstream example for a specific 4xGB200
configuration and are **not** Xinference defaults:

.. code-block:: bash

   --kv-cache-dtype fp8 \
   --block-size 256 \
   --data-parallel-size 4 \
   --enable-expert-parallel \
   --moe-backend deep_gemm_mega_moe \
   --attention-config '{"use_fp4_indexer_cache": true}'
