.. _models_builtin_qwen_chat:

=========
Qwen Chat
=========

- **Model Name:** qwen-chat
- **Languages:** en, zh
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen-7B-Chat

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit and 8-bit quantization are not supported on macOS.
