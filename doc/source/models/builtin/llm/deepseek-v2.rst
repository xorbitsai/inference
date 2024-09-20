.. _models_llm_deepseek-v2:

========================================
deepseek-v2
========================================

- **Context Length:** 128000
- **Model Name:** deepseek-v2
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** deepseek-ai/DeepSeek-V2-Lite
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v2 --size-in-billions 16 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 236 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 236
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** deepseek-ai/DeepSeek-V2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V2>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v2 --size-in-billions 236 --model-format pytorch --quantization ${quantization}

