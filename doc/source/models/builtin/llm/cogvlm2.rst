.. _models_llm_cogvlm2:

========================================
cogvlm2
========================================

- **Context Length:** 8192
- **Model Name:** cogvlm2
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** CogVLM2 have achieved good results in many lists compared to the previous generation of CogVLM open source models. Its excellent performance can compete with some non-open source models.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 20 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 20
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** THUDM/cogvlm2-llama3-chinese-chat-19B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name cogvlm2 --size-in-billions 20 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 20 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 20
- **Quantizations:** int4
- **Engines**: Transformers
- **Model ID:** THUDM/cogvlm2-llama3-chinese-chat-19B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/cogvlm2-llama3-chinese-chat-19B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/cogvlm2-llama3-chinese-chat-19B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name cogvlm2 --size-in-billions 20 --model-format pytorch --quantization ${quantization}

