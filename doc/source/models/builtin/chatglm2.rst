.. _models_builtin_chatglm2:

========
ChatGLM2
========

- **Model Name:** chatglm2
- **Languages:** en, zh
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (ggmlv3, 6 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 6
- **Quantizations:** q4_0, q4_1, q5_0, q5_1, q8_0
- **Model ID:** Xorbits/chatglm2-6B-GGML

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name chatglm2 --size-in-billions 6 --model-format ggmlv3 --quantization ${quantization}

Model Spec 2 (pytorch, 6 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** THUDM/chatglm2-6b

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name chatglm2 --size-in-billions 6 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
