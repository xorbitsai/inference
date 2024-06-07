.. _models_llm_glm-4v:

========================================
glm-4v
========================================

- **Context Length:** 8192
- **Model Name:** glm-4v
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** THUDM/glm-4v-9b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-4v-9b>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-4v-9b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4v --size-in-billions 9 --model-format pytorch --quantization ${quantization}

