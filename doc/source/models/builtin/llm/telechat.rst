.. _models_llm_telechat:

========================================
telechat
========================================

- **Context Length:** 8192
- **Model Name:** telechat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** The TeleChat is a large language model developed and trained by China Telecom Artificial Intelligence Technology Co., LTD. The 7B model base is trained with 1.5 trillion Tokens and 3 trillion Tokens and Chinese high-quality corpus.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** Tele-AI/telechat-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Tele-AI/telechat-7B>`__, `ModelScope <https://modelscope.cn/models/TeleAI/telechat-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name telechat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** int4, int8
- **Engines**: Transformers
- **Model ID:** Tele-AI/telechat-7B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Tele-AI/telechat-7B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/TeleAI/telechat-7B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name telechat --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 3 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 12
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** Tele-AI/TeleChat-12B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Tele-AI/TeleChat-12B>`__, `ModelScope <https://modelscope.cn/models/Tele-AI/TeleChat-12B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name telechat --size-in-billions 12 --model-format pytorch --quantization ${quantization}


Model Spec 4 (gptq, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 12
- **Quantizations:** int4, int8
- **Engines**: Transformers
- **Model ID:** Tele-AI/TeleChat-12B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Tele-AI/TeleChat-12B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/Tele-AI/TeleChat-12B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name telechat --size-in-billions 12 --model-format gptq --quantization ${quantization}


Model Spec 5 (pytorch, 52 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 52
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** Tele-AI/TeleChat-52B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Tele-AI/TeleChat-52B>`__, `ModelScope <https://modelscope.cn/models/Tele-AI/TeleChat-52B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name telechat --size-in-billions 52 --model-format pytorch --quantization ${quantization}

