.. _models_builtin_baichuan_chat:

=============
Baichuan Chat
=============

- **Model Name:** baichuan-chat
- **Languages:** en, zh
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan-13B-Chat

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   Not supported on macOS.
