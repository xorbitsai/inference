.. _models_llm_zephyr-7b-beta:

========================================
zephyr-7b-beta
========================================

- **Context Length:** 8192
- **Model Name:** zephyr-7b-beta
- **Languages:** en
- **Abilities:** chat
- **Description:** Zephyr-7B-β is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** HuggingFaceH4/zephyr-7b-beta
- **Model Hubs**:  `Hugging Face <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`__, `ModelScope <https://modelscope.cn/models/modelscope/zephyr-7b-beta>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name zephyr-7b-beta --size-in-billions 7 --model-format pytorch --quantization ${quantization}

