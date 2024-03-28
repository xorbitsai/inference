.. _models_llm_deepseek-vl-chat:

========================================
deepseek-vl-chat
========================================

- **Context Length:** 4096
- **Model Name:** deepseek-vl-chat
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** DeepSeek-VL possesses general multimodal understanding capabilities, capable of processing logical diagrams, web pages, formula recognition, scientific literature, natural images, and embodied intelligence in complex scenarios.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_3
- **Quantizations:** none
- **Model ID:** deepseek-ai/deepseek-vl-1.3b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-vl-1.3b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-vl-chat --size-in-billions 1_3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Model ID:** deepseek-ai/deepseek-vl-7b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-vl-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}

