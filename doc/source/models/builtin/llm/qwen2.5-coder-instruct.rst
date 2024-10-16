.. _models_llm_qwen2.5-coder-instruct:

========================================
qwen2.5-coder-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2.5-coder-instruct
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen).

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

