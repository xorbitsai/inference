.. _models_llm_glm-4.7-flash:

========================================
GLM-4.7-Flash
========================================

- **Context Length:** 202752
- **Model Name:** GLM-4.7-Flash
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** GLM-4.7-Flash is a 30B-A3B MoE model. As the strongest model in the 30B class, it offers a lightweight deployment option that balances performance and efficiency.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** zai-org/GLM-4.7-Flash
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.7-Flash>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.7-Flash>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name GLM-4.7-Flash --size-in-billions 30 --model-format pytorch --quantization ${quantization}

