.. _models_llm_qwen3-instruct:

========================================
Qwen3-Instruct
========================================

- **Context Length:** 262144
- **Model Name:** Qwen3-Instruct
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** We introduce the updated version of the Qwen3-235B-A22B non-thinking mode, named Qwen3-235B-A22B-Instruct-2507

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 235
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-235B-A22B-Instruct-2507
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 235 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-30B-A3B-Instruct-2507
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-4B-Instruct-2507
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 4 (fp8, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 235
- **Quantizations:** fp8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 235 --model-format fp8 --quantization ${quantization}


Model Spec 5 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 6 (fp8, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3-4B-Instruct-2507-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 4 --model-format fp8 --quantization ${quantization}


Model Spec 7 (gptq, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 235
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-235B-A22B-Instruct-2507-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-235B-A22B-Instruct-2507-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-235B-A22B-Instruct-2507-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 235 --model-format gptq --quantization ${quantization}


Model Spec 8 (gptq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 30
- **Quantizations:** Int8
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 30 --model-format gptq --quantization ${quantization}


Model Spec 9 (gptq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 4
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers
- **Model ID:** JunHowie/Qwen3-4B-Instruct-2507-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Qwen3-4B-Instruct-2507-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Qwen3-4B-Instruct-2507-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 4 --model-format gptq --quantization ${quantization}


Model Spec 10 (awq, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 235
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-235B-A22B-Instruct-2507-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 235 --model-format awq --quantization ${quantization}


Model Spec 11 (awq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 30
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-30B-A3B-Instruct-2507-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 30 --model-format awq --quantization ${quantization}


Model Spec 12 (awq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 4
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** Eslzzyl/Qwen3-4B-Instruct-2507-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Eslzzyl/Qwen3-4B-Instruct-2507-AWQ>`__, `ModelScope <https://modelscope.cn/models/Eslzzyl/Qwen3-4B-Instruct-2507-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 4 --model-format awq --quantization ${quantization}


Model Spec 13 (mlx, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 235
- **Quantizations:** 3bit, 4bit, 5bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-235B-A22B-Instruct-2507-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-235B-A22B-Instruct-2507-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-235B-A22B-Instruct-2507-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 235 --model-format mlx --quantization ${quantization}


Model Spec 14 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 5bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-30B-A3B-Instruct-2507-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-30B-A3B-Instruct-2507-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 15 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 4bit, 5bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-4B-Instruct-2507-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-4B-Instruct-2507-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 16 (ggufv2, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 235
- **Quantizations:** BF16, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 235 --model-format ggufv2 --quantization ${quantization}


Model Spec 17 (ggufv2, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 30
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, UD-TQ1_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 30 --model-format ggufv2 --quantization ${quantization}


Model Spec 18 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-4B-Instruct-2507-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-4B-Instruct-2507-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Instruct --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}

