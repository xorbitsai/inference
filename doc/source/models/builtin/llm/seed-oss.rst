.. _models_llm_seed-oss:

========================================
seed-oss
========================================

- **Context Length:** 524288
- **Model Name:** seed-oss
- **Languages:** en, zh
- **Abilities:** chat, reasoning, tools
- **Description:** Seed-OSS is a series of open-source large language models developed by ByteDance's Seed Team, designed for powerful long-context, reasoning, agent and general capabilities, and versatile developer-friendly features. Although trained with only 12T tokens, Seed-OSS achieves excellent performance on several popular open benchmarks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 36 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 36
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** ByteDance-Seed/Seed-OSS-36B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct>`__, `ModelScope <https://modelscope.cn/models/ByteDance-Seed/Seed-OSS-36B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seed-oss --size-in-billions 36 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 36 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 36
- **Quantizations:** Int8, Int4, Int3
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Seed-OSS-36B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Seed-OSS-36B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/tclf90/Seed-OSS-36B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seed-oss --size-in-billions 36 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 36 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 36
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Seed-OSS-36B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Seed-OSS-36B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Seed-OSS-36B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seed-oss --size-in-billions 36 --model-format awq --quantization ${quantization}


Model Spec 4 (mlx, 36 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 36
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/Seed-OSS-36B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Seed-OSS-36B-Instruct-4bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Seed-OSS-36B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seed-oss --size-in-billions 36 --model-format mlx --quantization ${quantization}


Model Spec 5 (ggufv2, 36 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 36
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Seed-OSS-36B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Seed-OSS-36B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Seed-OSS-36B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seed-oss --size-in-billions 36 --model-format ggufv2 --quantization ${quantization}

