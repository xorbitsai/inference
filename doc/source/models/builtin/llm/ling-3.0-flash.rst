.. _models_llm_ling-3.0-flash:

========================================
Ling-3.0-flash
========================================

- **Context Length:** 262144
- **Model Name:** Ling-3.0-flash
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning, hybrid
- **Description:** Ling-3.0-flash is a native hybrid-linear reasoning MoE model with 124B total parameters and 5.1B activated parameters per token.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 124 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 124
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** inclusionAI/Ling-3.0-flash
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-flash>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-flash>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-flash --size-in-billions 124 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 124 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 124
- **Quantizations:** FP8
- **Engines**: vLLM, Transformers
- **Model ID:** inclusionAI/Ling-3.0-flash-fp8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-flash-fp8>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-fp8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-flash --size-in-billions 124 --model-format fp8 --quantization ${quantization}


Model Spec 3 (fp4, 124 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp4
- **Model Size (in billions):** 124
- **Quantizations:** FP4
- **Engines**: vLLM
- **Model ID:** inclusionAI/Ling-3.0-flash-fp4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-fp4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-flash --size-in-billions 124 --model-format fp4 --quantization ${quantization}


Model Spec 4 (pytorch, 124 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 124
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** inclusionAI/Ling-3.0-flash-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-flash-int4>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-flash --size-in-billions 124 --model-format pytorch --quantization ${quantization}


Model Spec 5 (ggufv2, 124 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 124
- **Quantizations:** BF16, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** inclusionAI/Ling-3.0-flash-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-flash-GGUF>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-flash-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-flash --size-in-billions 124 --model-format ggufv2 --quantization ${quantization}

