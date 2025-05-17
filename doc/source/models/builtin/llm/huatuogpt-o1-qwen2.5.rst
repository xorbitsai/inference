.. _models_llm_huatuogpt-o1-qwen2.5:

========================================
HuatuoGPT-o1-Qwen2.5
========================================

- **Context Length:** 32768
- **Model Name:** HuatuoGPT-o1-Qwen2.5
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** HuatuoGPT-o1 is a medical LLM designed for advanced medical reasoning. It generates a complex thought process, reflecting and refining its reasoning, before providing a final response.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** FreedomIntelligence/HuatuoGPT-o1-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B>`__, `ModelScope <https://modelscope.cn/models/FreedomIntelligence/HuatuoGPT-o1-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name HuatuoGPT-o1-Qwen2.5 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** FreedomIntelligence/HuatuoGPT-o1-72B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-72B>`__, `ModelScope <https://modelscope.cn/models/FreedomIntelligence/HuatuoGPT-o1-72B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name HuatuoGPT-o1-Qwen2.5 --size-in-billions 72 --model-format pytorch --quantization ${quantization}

