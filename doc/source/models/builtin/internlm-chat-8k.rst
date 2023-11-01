.. _models_builtin_internlm_chat_8k:


================
InternLM Chat 8K
================

- **Model Name:** internlm-chat-8k
- **Languages:** en, zh
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** internlm/internlm-chat-7b-8k

Execute the following command to launch the model, remember to replace ``${quantization}`` with your chosen quantization method from the options listed above::

   xinference launch --model-name internlm-chat-8k --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.