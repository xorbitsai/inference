.. _models_llm_xverse-chat:

========================================
xverse-chat
========================================

- **Context Length:** 2048
- **Model Name:** xverse-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** XVERSEB-Chat is the aligned version of model XVERSE.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** xverse/XVERSE-7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-7B-Chat>`_, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-7B-Chat>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name xverse-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** xverse/XVERSE-13B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-13B-Chat>`_, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-13B-Chat>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name xverse-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}

