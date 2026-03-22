.. _models_llm_qwen3.5:

========================================
qwen3.5
========================================

- **Context Length:** 262144
- **Model Name:** qwen3.5
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Over recent months, we have intensified our focus on developing foundation models that deliver exceptional utility and performance. Qwen3.5 represents a significant leap forward, integrating breakthroughs in multimodal learning, architectural efficiency, reinforcement learning scale, and global accessibility to empower developers and enterprises with unprecedented capability and efficiency

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 397
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-397B-A17B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 397
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3.5-397B-A17B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format fp8 --quantization ${quantization}


Model Spec 3 (gptq, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 397
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format gptq --quantization ${quantization}


Model Spec 4 (awq, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 397
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3.5-397B-A17B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.5-397B-A17B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.5-397B-A17B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format awq --quantization ${quantization}


Model Spec 5 (ggufv2, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 397
- **Quantizations:** UD-TQ1_0
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-397B-A17B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-397B-A17B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (mlx, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 397
- **Quantizations:** 4bit, 5bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-397B-A17B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-397B-A17B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format mlx --quantization ${quantization}


Model Spec 7 (pytorch, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 122
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-122B-A10B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-122B-A10B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format pytorch --quantization ${quantization}


Model Spec 8 (fp8, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 122
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3.5-122B-A10B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format fp8 --quantization ${quantization}


Model Spec 9 (gptq, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 122
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format gptq --quantization ${quantization}


Model Spec 10 (awq, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 122
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3.5-122B-A10B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.5-122B-A10B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.5-122B-A10B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format awq --quantization ${quantization}


Model Spec 11 (ggufv2, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 122
- **Quantizations:** UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-122B-A10B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-122B-A10B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format ggufv2 --quantization ${quantization}


Model Spec 12 (mlx, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 122
- **Quantizations:** 4bit, 5bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-122B-A10B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-122B-A10B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format mlx --quantization ${quantization}


Model Spec 13 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-35B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 14 (fp8, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 35
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3.5-35B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format fp8 --quantization ${quantization}


Model Spec 15 (gptq, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 35
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format gptq --quantization ${quantization}


Model Spec 16 (awq, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 35
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3.5-35B-A3B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.5-35B-A3B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format awq --quantization ${quantization}


Model Spec 17 (ggufv2, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 35
- **Quantizations:** MXFP4_MOE, Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ1_M, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_L, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_S, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-35B-A3B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-35B-A3B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format ggufv2 --quantization ${quantization}


Model Spec 18 (mlx, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 35
- **Quantizations:** 4bit, 5bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-35B-A3B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-35B-A3B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format mlx --quantization ${quantization}


Model Spec 19 (pytorch, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 27
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-27B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-27B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-27B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format pytorch --quantization ${quantization}


Model Spec 20 (fp8, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 27
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3.5-27B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-27B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-27B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format fp8 --quantization ${quantization}


Model Spec 21 (gptq, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 27
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-27B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-27b-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format gptq --quantization ${quantization}


Model Spec 22 (awq, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 27
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3.5-27B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.5-27B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.5-27B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format awq --quantization ${quantization}


Model Spec 23 (ggufv2, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 27
- **Quantizations:** IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-27B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-27B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-27B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format ggufv2 --quantization ${quantization}


Model Spec 24 (mlx, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 27
- **Quantizations:** 4bit, 5bit, 6bit, 8bit, bf16, mxfp8
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-27B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-27B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-27B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format mlx --quantization ${quantization}


Model Spec 25 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-9B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-9B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-9B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 26 (awq, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 9
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** QuantTrio/Qwen3.5-9B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.5-9B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 9 --model-format awq --quantization ${quantization}


Model Spec 27 (ggufv2, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 9
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-9B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-9B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-9B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 9 --model-format ggufv2 --quantization ${quantization}


Model Spec 28 (mlx, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 9
- **Quantizations:** 4bit, 5bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-9B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-9B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-9B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 9 --model-format mlx --quantization ${quantization}


Model Spec 29 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-4B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 30 (awq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 4
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** QuantTrio/Qwen3.5-4B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.5-4B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.5-4B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 4 --model-format awq --quantization ${quantization}


Model Spec 31 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-4B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-4B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-4B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}


Model Spec 32 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 3bit, 4bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-4B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-4B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-4B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 33 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-2B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-2B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-2B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 34 (awq, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 2
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** QuantTrio/Qwen3.5-2B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.5-2B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.5-2B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 2 --model-format awq --quantization ${quantization}


Model Spec 35 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-2B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-2B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}


Model Spec 36 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** 3bit, 4bit, 5bit, 6bit, 8bit, bf16, mxfp4, mxfp8, nvfp4
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-2B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-2B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-2B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 37 (pytorch, 0_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_8
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3.5-0.8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-0.8B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-0.8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 0_8 --model-format pytorch --quantization ${quantization}


Model Spec 38 (ggufv2, 0_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 0_8
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.5-0.8B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.5-0.8B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 0_8 --model-format ggufv2 --quantization ${quantization}


Model Spec 39 (mlx, 0_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 0_8
- **Quantizations:** 3bit, 4bit, 5bit, 6bit, 8bit, bf16, mxfp4, mxfp8, nvfp4
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.5-0.8B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.5-0.8B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.5-0.8B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 0_8 --model-format mlx --quantization ${quantization}

