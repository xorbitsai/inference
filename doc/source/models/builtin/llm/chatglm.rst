.. _models_llm_chatglm:

========================================
chatglm
========================================

- **Context Length:** 2048
- **Model Name:** chatglm
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** ChatGLM is an open-source General Language Model (GLM) based LLM trained on both Chinese and English data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** THUDM/chatglm-6b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/chatglm-6b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name chatglm --size-in-billions 6 --model-format pytorch --quantization ${quantization}

