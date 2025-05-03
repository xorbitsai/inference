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
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** xverse/XVERSE-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-7B>`__, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name xverse --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** xverse/XVERSE-13B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-13B>`__, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-13B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name xverse --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 65 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 65
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** xverse/XVERSE-65B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/xverse/XVERSE-65B>`__, `ModelScope <https://modelscope.cn/models/xverse/XVERSE-65B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name xverse --size-in-billions 65 --model-format pytorch --quantization ${quantization}

