.. _models_llm_deepseek-v3.2:

========================================
DeepSeek-V3.2
========================================

- **Context Length:** 131072
- **Model Name:** DeepSeek-V3.2
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 671
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** deepseek-ai/DeepSeek-V3.2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V3.2>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V3.2 --size-in-billions 671 --model-format pytorch --quantization ${quantization}

