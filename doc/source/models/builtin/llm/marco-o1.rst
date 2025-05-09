.. _models_llm_marco-o1:

========================================
marco-o1
========================================

- **Context Length:** 32768
- **Model Name:** marco-o1
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** AIDC-AI/Marco-o1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Marco-o1>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Marco-o1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name marco-o1 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** QuantFactory/Marco-o1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantFactory/Marco-o1-GGUF>`__, `ModelScope <https://modelscope.cn/models/QuantFactory/Marco-o1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name marco-o1 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

