.. _models_llm_glm4-chat:

========================================
glm4-chat
========================================

- **Context Length:** 131072
- **Model Name:** glm4-chat
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** THUDM/glm-4-9b-chat-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-4-9b-chat-hf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-hf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm4-chat --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 9
- **Quantizations:** Q2_K, IQ3_XS, IQ3_S, IQ3_M, Q3_K_S, Q3_K_L, Q3_K, IQ4_XS, IQ4_NL, Q4_K_S, Q4_K, Q5_K_S, Q5_K, Q6_K, Q8_0, BF16, FP16
- **Engines**: vLLM, llama.cpp
- **Model ID:** legraphista/glm-4-9b-chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/legraphista/glm-4-9b-chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/glm-4-9b-chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm4-chat --size-in-billions 9 --model-format ggufv2 --quantization ${quantization}

