.. _models_llm_kimi-k3:

========================================
Kimi-K3
========================================

- **Context Length:** 1048576
- **Model Name:** Kimi-K3
- **Languages:** en, zh
- **Abilities:** chat, vision, reasoning, tools
- **Description:** Kimi K3 is an open-weight, native multimodal agentic model and our most capable model to date.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2779_93 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2779_93
- **Quantizations:** none
- **Engines**: vLLM
- **Model ID:** moonshotai/Kimi-K3
- **Model Hubs**:  `Hugging Face <https://huggingface.co/moonshotai/Kimi-K3>`__, `ModelScope <https://modelscope.cn/models/moonshotai/Kimi-K3>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Kimi-K3 --size-in-billions 2779_93 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 2779_93 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2779_93
- **Quantizations:** UD-IQ1_M, UD-IQ1_S, UD-IQ2_XXS, UD-Q1_0, UD-Q2_K_XL, UD-Q4_K_XL, UD-Q8_K_XL, UD-TQ1_0, UD-TQ2_0
- **Engines**: llama.cpp
- **Model ID:** unsloth/Kimi-K3-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Kimi-K3-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Kimi-K3-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Kimi-K3 --size-in-billions 2779_93 --model-format ggufv2 --quantization ${quantization}
