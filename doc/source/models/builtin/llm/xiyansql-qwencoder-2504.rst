.. _models_llm_xiyansql-qwencoder-2504:

========================================
XiYanSQL-QwenCoder-2504
========================================

- **Context Length:** 32768
- **Model Name:** XiYanSQL-QwenCoder-2504
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** The XiYanSQL-QwenCoder models, as multi-dialect SQL base models, demonstrating robust SQL generation capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** XGenerationLab/XiYanSQL-QwenCoder-7B-2504
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-7B-2504>`__, `ModelScope <https://modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-7B-2504>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name XiYanSQL-QwenCoder-2504 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** XGenerationLab/XiYanSQL-QwenCoder-32B-2504
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-32B-2504>`__, `ModelScope <https://modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-32B-2504>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name XiYanSQL-QwenCoder-2504 --size-in-billions 32 --model-format pytorch --quantization ${quantization}

