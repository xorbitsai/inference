.. _models_llm_internlm3-instruct:

========================================
internlm3-instruct
========================================

- **Context Length:** 32768
- **Model Name:** internlm3-instruct
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** InternLM3 has open-sourced an 8-billion parameter instruction model, InternLM3-8B-Instruct, designed for general-purpose usage and advanced reasoning.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** internlm/internlm3-8b-instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm3-8b-instruct>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm3-instruct --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** internlm/internlm3-8b-instruct-gptq-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm3-8b-instruct-gptq-int4>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct-gptq-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm3-instruct --size-in-billions 8 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** internlm/internlm3-8b-instruct-awq
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm3-8b-instruct-awq>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm3-instruct --size-in-billions 8 --model-format awq --quantization ${quantization}


Model Spec 4 (ggufv2, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 8
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** internlm/internlm3-8b-instruct-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm3-8b-instruct-gguf>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm3-instruct --size-in-billions 8 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/internlm3-8b-instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/internlm3-8b-instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/internlm3-8b-instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm3-instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}

