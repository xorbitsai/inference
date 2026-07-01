.. _models_llm_baichuan-m2:

========================================
Baichuan-M2
========================================

- **Context Length:** 131072
- **Model Name:** Baichuan-M2
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** Baichuan-M2-32B is Baichuan AI's medical-enhanced reasoning model, the second medical model released by Baichuan. Designed for real-world medical reasoning tasks, this model builds upon Qwen2.5-32B with an innovative Large Verifier System. Through domain-specific fine-tuning on real-world medical questions, it achieves breakthrough medical performance while maintaining strong general capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** baichuan-inc/Baichuan-M2-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan-M2-32B>`__, `ModelScope <https://modelscope.cn/models/baichuan-inc/Baichuan-M2-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Baichuan-M2 --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** baichuan-inc/Baichuan-M2-32B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan-M2-32B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/baichuan-inc/Baichuan-M2-32B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Baichuan-M2 --size-in-billions 32 --model-format gptq --quantization ${quantization}

