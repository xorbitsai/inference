.. _models_llm_qwen2.5-omni:

========================================
qwen2.5-omni
========================================

- **Context Length:** 32768
- **Model Name:** qwen2.5-omni
- **Languages:** en, zh
- **Abilities:** chat, vision, audio, omni
- **Description:** Qwen2.5-Omni: the new flagship end-to-end multimodal model in the Qwen series.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2.5-Omni-3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Omni-3B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen2.5-Omni-3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-omni --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2.5-Omni-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Omni-7B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-omni --size-in-billions 7 --model-format pytorch --quantization ${quantization}

