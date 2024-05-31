.. _models_llm_qwen-vl-chat:

========================================
qwen-vl-chat
========================================

- **Context Length:** 4096
- **Model Name:** qwen-vl-chat
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** Qwen-VL-Chat supports more flexible interaction, such as multiple image inputs, multi-round question answering, and creative capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen-VL-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-VL-Chat>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen-VL-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen-vl-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen-VL-Chat-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-VL-Chat-{quantization}>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen-VL-Chat-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen-vl-chat --size-in-billions 7 --model-format gptq --quantization ${quantization}

