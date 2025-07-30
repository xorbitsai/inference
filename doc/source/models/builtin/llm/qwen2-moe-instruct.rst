.. _models_llm_qwen2-moe-instruct:

========================================
qwen2-moe-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2-moe-instruct
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Qwen2 is the new series of Qwen large language models. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-57B-A14B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-57B-A14B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-moe-instruct --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-moe-instruct --size-in-billions 14 --model-format gptq --quantization ${quantization}


Model Spec 3 (ggufv2, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 14
- **Quantizations:** q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: vLLM, llama.cpp
- **Model ID:** Qwen/Qwen2-57B-A14B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-57B-A14B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-moe-instruct --size-in-billions 14 --model-format ggufv2 --quantization ${quantization}

