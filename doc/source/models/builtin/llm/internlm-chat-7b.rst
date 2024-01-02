.. _models_llm_internlm-chat-7b:

========================================
internlm-chat-7b
========================================

- **Context Length:** 4096
- **Model Name:** internlm-chat-7b
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Internlm-chat is a fine-tuned version of the Internlm LLM, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** internlm/internlm-chat-7b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm-chat-7b>`_, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name internlm-chat-7b --size-in-billions 7 --model-format pytorch --quantization ${quantization}

