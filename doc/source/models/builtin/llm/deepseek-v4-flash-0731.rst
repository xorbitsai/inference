.. _models_llm_deepseek-v4-flash-0731:

========================================
DeepSeek-V4-Flash-0731
========================================

- **Context Length:** 1048576
- **Model Name:** DeepSeek-V4-Flash-0731
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** Official DeepSeek-V4-Flash release with enhanced agentic capabilities and an attached DSpark speculative decoding module.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (fp8, 304 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 304
- **Quantizations:** fp8
- **Engines**: vLLM
- **Model ID:** deepseek-ai/DeepSeek-V4-Flash-0731
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4-Flash-0731 --size-in-billions 304 --model-format fp8 --quantization ${quantization}

