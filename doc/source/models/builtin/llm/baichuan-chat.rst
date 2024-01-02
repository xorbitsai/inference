.. _models_llm_baichuan-chat:

========================================
baichuan-chat
========================================

- **Context Length:** 4096
- **Model Name:** baichuan-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Baichuan-chat is a fine-tuned version of the Baichuan LLM, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan-13B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan-13B-Chat>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}

