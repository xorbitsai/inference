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
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** xverse/XVERSE-7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-7B-Chat>`__, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-7B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name xverse-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** xverse/XVERSE-13B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-13B-Chat>`__, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-13B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name xverse-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}

