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
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/DeepSeek-V4.1-Flash-1017
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4.1-Flash-1017 --size-in-billions 552 --model-format fp8 --quantization ${quantization}


Model Spec 2 (ggufv2, 552 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 552
- **Quantizations:** UD-IQ1_M, UD-IQ1_S, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/DeepSeek-V4.1-Flash-1017-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4.1-Flash-1017 --size-in-billions 552 --model-format ggufv2 --quantization ${quantization}

