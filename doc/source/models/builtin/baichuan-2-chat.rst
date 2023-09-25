.. _models_builtin_baichuan_2_chat:

===============
Baichuan-2-Chat
===============

- **Context Length:** 4096
- **Model Name:** baichuan-2-chat
- **Languages:** en, zh
- **Abilities:** embed, generate, chat
- **Description:** Baichuan2-chat is a fine-tuned version of the Baichuan LLM, specializing in chatting.

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan2-7B-Chat
- **Model Revision:** 2ce891951e000c36c65442608a0b95fd09b405dc

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan-2-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   Not supported on macOS.


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan2-13B-Chat
- **Model Revision:** a56c793eb7a721ab6c270f779024e0375e8afd4a

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan-2-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   Not supported on macOS.
