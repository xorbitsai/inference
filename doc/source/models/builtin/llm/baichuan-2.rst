.. _models_llm_baichuan-2:

========================================
baichuan-2
========================================

- **Context Length:** 4096
- **Model Name:** baichuan-2
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Baichuan2 is an open-source Transformer based LLM that is trained on both Chinese and English data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** baichuan-inc/Baichuan2-7B-Base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan2-7B-Base>`__, `ModelScope <https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name baichuan-2 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** baichuan-inc/Baichuan2-13B-Base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan2-13B-Base>`__, `ModelScope <https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name baichuan-2 --size-in-billions 13 --model-format pytorch --quantization ${quantization}

