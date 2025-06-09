.. _models_llm_huatuogpt-o1-llama-3.1:

========================================
HuatuoGPT-o1-LLaMA-3.1
========================================

- **Context Length:** 131072
- **Model Name:** HuatuoGPT-o1-LLaMA-3.1
- **Languages:** en
- **Abilities:** chat, tools
- **Description:** HuatuoGPT-o1 is a medical LLM designed for advanced medical reasoning. It generates a complex thought process, reflecting and refining its reasoning, before providing a final response.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** FreedomIntelligence/HuatuoGPT-o1-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-8B>`__, `ModelScope <https://modelscope.cn/models/FreedomIntelligence/HuatuoGPT-o1-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name HuatuoGPT-o1-LLaMA-3.1 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** FreedomIntelligence/HuatuoGPT-o1-70B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-70B>`__, `ModelScope <https://modelscope.cn/models/FreedomIntelligence/HuatuoGPT-o1-70B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name HuatuoGPT-o1-LLaMA-3.1 --size-in-billions 70 --model-format pytorch --quantization ${quantization}

