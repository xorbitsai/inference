.. _models_llm_qwen3:

========================================
qwen3
========================================

- **Context Length:** 40960
- **Model Name:** qwen3
- **Languages:** en, zh
- **Abilities:** chat, reasoning, tools
- **Description:** Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_6
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-0.6B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-0.6B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-0.6B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 0_6
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-0.6B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-0.6B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-0.6B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format fp8 --quantization ${quantization}


Model Spec 3 (gptq, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 0_6
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** JunHowie/Qwen3-0.6B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-0.6B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-0.6B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format gptq --quantization ${quantization}


Model Spec 4 (mlx, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 0_6
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-0.6B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-0.6B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-0.6B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format mlx --quantization ${quantization}


Model Spec 5 (ggufv2, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 0_6
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-0.6B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-0.6B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (pytorch, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_7
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-1.7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-1.7B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-1.7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format pytorch --quantization ${quantization}


Model Spec 7 (fp8, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 1_7
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-1.7B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-1.7B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-1.7B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format fp8 --quantization ${quantization}


Model Spec 8 (gptq, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1_7
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** JunHowie/Qwen3-1.7B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-1.7B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-1.7B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format gptq --quantization ${quantization}


Model Spec 9 (mlx, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1_7
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-1.7B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-1.7B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-1.7B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format mlx --quantization ${quantization}


Model Spec 10 (ggufv2, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_7
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-1.7B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-1.7B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format ggufv2 --quantization ${quantization}


Model Spec 11 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-4B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 12 (fp8, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 4
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-4B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-4B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-4B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format fp8 --quantization ${quantization}


Model Spec 13 (gptq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 4
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** JunHowie/Qwen3-4B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-4B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-4B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format gptq --quantization ${quantization}


Model Spec 14 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-4B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-4B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-4B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 15 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-4B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-4B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-4B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}


Model Spec 16 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-8B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 17 (fp8, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 8
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-8B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-8B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-8B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format fp8 --quantization ${quantization}


Model Spec 18 (gptq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 8
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** JunHowie/Qwen3-8B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-8B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-8B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format gptq --quantization ${quantization}


Model Spec 19 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-8B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-8B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-8B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 20 (ggufv2, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 8
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-8B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-8B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-8B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format ggufv2 --quantization ${quantization}


Model Spec 21 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-14B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-14B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-14B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 22 (fp8, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 14
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-14B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-14B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-14B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format fp8 --quantization ${quantization}


Model Spec 23 (awq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-14B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-14B-AWQ>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-14B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format awq --quantization ${quantization}


Model Spec 24 (gptq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** JunHowie/Qwen3-14B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-14B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-14B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format gptq --quantization ${quantization}


Model Spec 25 (mlx, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 14
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-14B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-14B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-14B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format mlx --quantization ${quantization}


Model Spec 26 (ggufv2, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 14
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-14B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-14B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-14B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format ggufv2 --quantization ${quantization}


Model Spec 27 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-30B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-30B-A3B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-30B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 28 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-30B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 29 (gptq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 30
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** JunHowie/Qwen3-30B-A3B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-30B-A3B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-30B-A3B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format gptq --quantization ${quantization}


Model Spec 30 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-30B-A3B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-30B-A3B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-30B-A3B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 31 (ggufv2, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 30
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-30B-A3B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-30B-A3B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format ggufv2 --quantization ${quantization}


Model Spec 32 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-32B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 33 (fp8, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 32
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-32B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-32B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-32B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format fp8 --quantization ${quantization}


Model Spec 34 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-32B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-32B-AWQ>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-32B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format awq --quantization ${quantization}


Model Spec 35 (gptq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 32
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** JunHowie/Qwen3-32B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-32B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-32B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format gptq --quantization ${quantization}


Model Spec 36 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-32B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-32B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-32B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format mlx --quantization ${quantization}


Model Spec 37 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-32B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-32B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-32B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 38 (pytorch, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 235
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-235B-A22B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-235B-A22B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-235B-A22B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 235 --model-format pytorch --quantization ${quantization}


Model Spec 39 (fp8, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 235
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-235B-A22B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-235B-A22B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 235 --model-format fp8 --quantization ${quantization}


Model Spec 40 (mlx, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 235
- **Quantizations:** 3bit, 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen/Qwen3-235B-A22B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen/Qwen3-235B-A22B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-235B-A22B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 235 --model-format mlx --quantization ${quantization}


Model Spec 41 (ggufv2, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 235
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q5_K_M, Q6_K, Q8_0, BF16, UD-Q2_K_XL, UD-Q3_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-235B-A22B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-235B-A22B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 235 --model-format ggufv2 --quantization ${quantization}

