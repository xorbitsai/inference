.. _models_llm_zephyr-7b-alpha:

========================================
zephyr-7b-alpha
========================================

- **Context Length:** 8192
- **Model Name:** zephyr-7b-alpha
- **Languages:** en
- **Abilities:** chat
- **Description:** Zephyr-7B-α is the first model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** HuggingFaceH4/zephyr-7b-alpha
- **Model Hubs**:  `Hugging Face <https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha>`__, `ModelScope <https://modelscope.cn/models/keepitsimple/zephyr-7b-alpha>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name zephyr-7b-alpha --size-in-billions 7 --model-format pytorch --quantization ${quantization}

