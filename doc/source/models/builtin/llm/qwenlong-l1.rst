.. _models_llm_qwenlong-l1:

========================================
qwenLong-l1
========================================

- **Context Length:** 32768
- **Model Name:** qwenLong-l1
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Tongyi-Zhiwen/QwenLong-L1-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Tongyi-Zhiwen/QwenLong-L1-32B>`__, `ModelScope <https://modelscope.cn/models/iic/QwenLong-L1-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwenLong-l1 --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Tongyi-Zhiwen/QwenLong-L1-32B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Tongyi-Zhiwen/QwenLong-L1-32B-AWQ>`__, `ModelScope <https://modelscope.cn/models/iic/QwenLong-L1-32B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwenLong-l1 --size-in-billions 32 --model-format awq --quantization ${quantization}

