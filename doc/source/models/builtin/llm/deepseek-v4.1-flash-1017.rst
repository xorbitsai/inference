.. _models_llm_deepseek-v4.1-flash-1017:

========================================
DeepSeek-V4.1-Flash-1017
========================================

- **Context Length:** 1048576
- **Model Name:** DeepSeek-V4.1-Flash-1017
- **Languages:** en, zh
- **Abilities:** chat, vision, reasoning, hybrid, tools
- **Description:** We introduce DeepSeek-V4.1-Flash-1017, a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. The model natively processes images and text, and generates text autoregressively.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (fp8, 552 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 552
- **Quantizations:** fp8
- **Engines**: vLLM
- **Model ID:** deepseek-ai/DeepSeek-V4.1-Flash-1017
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4.1-Flash-1017 --size-in-billions 552 --model-format fp8 --quantization ${quantization}

