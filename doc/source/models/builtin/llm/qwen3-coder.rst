.. _models_llm_qwen3-coder:

========================================
Qwen3-Coder
========================================

- **Context Length:** 262144
- **Model Name:** Qwen3-Coder
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** we're announcing Qwen3-Coder, our most agentic code model to date

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 480 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 480
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-Coder-480B-A35B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-Coder-480B-A35B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 480 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-Coder-30B-A3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 3 (fp8, 480 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 480
- **Quantizations:** fp8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 480 --model-format fp8 --quantization ${quantization}


Model Spec 4 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 5 (gptq, 480 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 480
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-Coder-480B-A35B-Instruct-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-Coder-480B-A35B-Instruct-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-Coder-480B-A35B-Instruct-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 480 --model-format gptq --quantization ${quantization}


Model Spec 6 (gptq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 30
- **Quantizations:** Int8
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-Coder-30B-A3B-Instruct-GPTQ-Int8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 30 --model-format gptq --quantization ${quantization}


Model Spec 7 (awq, 480 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 480
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-Coder-480B-A35B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 480 --model-format awq --quantization ${quantization}


Model Spec 8 (awq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 30
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-Coder-30B-A3B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 30 --model-format awq --quantization ${quantization}


Model Spec 9 (mlx, 480 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 480
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-Coder-480B-A35B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-Coder-480B-A35B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-Coder-480B-A35B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 480 --model-format mlx --quantization ${quantization}


Model Spec 10 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** 3bit, 4bit, 5bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-Coder-30B-A3B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-Coder-30B-A3B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 11 (ggufv2, 480 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 480
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 480 --model-format ggufv2 --quantization ${quantization}


Model Spec 12 (ggufv2, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 30
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, UD-TQ1_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Coder --size-in-billions 30 --model-format ggufv2 --quantization ${quantization}

