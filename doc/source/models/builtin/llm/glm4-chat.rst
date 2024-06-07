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
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** THUDM/glm-4-9b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-4-9b-chat>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm4-chat --size-in-billions 9 --model-format pytorch --quantization ${quantization}

