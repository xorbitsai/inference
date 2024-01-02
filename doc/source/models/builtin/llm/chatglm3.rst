.. _models_llm_chatglm3:

========================================
chatglm3
========================================

- **Context Length:** 8192
- **Model Name:** chatglm3
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** ChatGLM3 is the third generation of ChatGLM, still open-source and trained on Chinese and English data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggmlv3, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 6
- **Quantizations:** q4_0
- **Model ID:** Xorbits/chatglm3-6B-GGML
- **Model hub**: 

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name chatglm3 --size-in-billions 6 --model-format ggmlv3 --quantization ${quantization}


Model Spec 2 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** THUDM/chatglm3-6b
- **Model hub**: 

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name chatglm3 --size-in-billions 6 --model-format pytorch --quantization ${quantization}

