.. _models_llm_qwen2.5-instruct-1m:

========================================
qwen2.5-instruct-1m
========================================

- **Context Length:** 1010000
- **Model Name:** qwen2.5-instruct-1m
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Qwen2.5-1M is the long-context version of the Qwen2.5 series models, supporting a context length of up to 1M tokens.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen2.5-7B-Instruct-1M
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct-1M>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct-1m --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen2.5-14B-Instruct-1M
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct-1M>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct-1m --size-in-billions 14 --model-format pytorch --quantization ${quantization}

