.. _models_llm_internlm2-chat:

========================================
internlm2-chat
========================================

- **Context Length:** 204800
- **Model Name:** internlm2-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** The second generation of the InternLM model, InternLM2.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Model ID:** internlm/internlm2-chat-7b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2-chat-7b>`_, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name internlm2-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 20 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 20
- **Quantizations:** none
- **Model ID:** internlm/internlm2-chat-20b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2-chat-20b>`_, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name internlm2-chat --size-in-billions 20 --model-format pytorch --quantization ${quantization}

