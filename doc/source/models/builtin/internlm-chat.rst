.. _models_builtin_internlm_chat:

=============
InternLM Chat
=============

- **Model Name:** internlm-chat
- **Languages:** en, zh
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** internlm/internlm-chat-7b

Execute the following command to launch the model, remember to replace `${quantization}` with your chosen quantization method from the options listed above::

   xinference launch --model-name internlm-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}
