.. _models_llm_cogvlm2-video-llama3-chat:

========================================
cogvlm2-video-llama3-chat
========================================

- **Context Length:** 8192
- **Model Name:** cogvlm2-video-llama3-chat
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** CogVLM2-Video achieves state-of-the-art performance on multiple video question answering tasks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 12
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** THUDM/cogvlm2-video-llama3-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/cogvlm2-video-llama3-chat>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name cogvlm2-video-llama3-chat --size-in-billions 12 --model-format pytorch --quantization ${quantization}

