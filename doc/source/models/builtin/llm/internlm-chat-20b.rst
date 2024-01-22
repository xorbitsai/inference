.. _models_llm_internlm-chat-20b:

========================================
internlm-chat-20b
========================================

- **Context Length:** 16384
- **Model Name:** internlm-chat-20b
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Pre-trained on over 2.3T Tokens containing high-quality English, Chinese, and code data. The Chat version has undergone SFT and RLHF training.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 20 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 20
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** internlm/internlm-chat-20b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm-chat-20b>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name internlm-chat-20b --size-in-billions 20 --model-format pytorch --quantization ${quantization}

