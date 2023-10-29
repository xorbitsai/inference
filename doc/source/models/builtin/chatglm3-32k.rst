.. _models_builtin_chatglm3_32k:


============
ChatGLM3-32K
============

- **Context Length:** 32768
- **Model Name:** chatglm3-32k
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** ChatGLM3 is the third generation of ChatGLM, still open-source and trained on Chinese and English data.

Specifications
^^^^^^^^^^^^^^

Model Spec (pytorch, 6 Billion)
+++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** THUDM/chatglm3-6b-32k

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name chatglm3-32k --size-in-billions 6 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
