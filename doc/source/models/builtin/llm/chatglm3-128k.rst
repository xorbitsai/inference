.. _models_llm_chatglm3-128k:

========================================
chatglm3-128k
========================================

- **Context Length:** 131072
- **Model Name:** chatglm3-128k
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** ChatGLM3 is the third generation of ChatGLM, still open-source and trained on Chinese and English data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** THUDM/chatglm3-6b-128k
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/chatglm3-6b-128k>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/chatglm3-6b-128k>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name chatglm3-128k --size-in-billions 6 --model-format pytorch --quantization ${quantization}

