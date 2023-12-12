.. _models_llm_chatglm2-32k:

========================================
chatglm2-32k
========================================

- **Context Length:** 32768
- **Model Name:** chatglm2-32k
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** ChatGLM2-32k is a special version of ChatGLM2, with a context window of 32k tokens instead of 8k.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** THUDM/chatglm2-6b-32k

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name chatglm2-32k --size-in-billions 6 --model-format pytorch --quantization ${quantization}

