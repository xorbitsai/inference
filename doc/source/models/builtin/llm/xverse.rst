.. _models_llm_xverse:

========================================
xverse
========================================

- **Context Length:** 2048
- **Model Name:** xverse
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** XVERSE is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** xverse/XVERSE-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-7B>`_, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-7B>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name xverse --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** xverse/XVERSE-13B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-13B>`_, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-13B>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name xverse --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 65 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 65
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** xverse/XVERSE-65B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-65B>`_, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-65B>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name xverse --size-in-billions 65 --model-format pytorch --quantization ${quantization}

