.. _models_llm_qwen3.6:

========================================
qwen3.6
========================================

- **Context Length:** 262144
- **Model Name:** qwen3.6
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Following the February release of the Qwen3.5 series, we're pleased to share the first open-weight variant of Qwen3.6. Built on direct feedback from the community, Qwen3.6 prioritizes stability and real-world utility, offering developers a more intuitive, responsive, and genuinely productive coding experience.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.6-35B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.6 --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 35
- **Quantizations:** FP8
- **Engines**: 
- **Model ID:** Qwen/Qwen3.6-35B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.6 --size-in-billions 35 --model-format fp8 --quantization ${quantization}


Model Spec 3 (awq, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 35
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** QuantTrio/Qwen3.6-35B-A3B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3.6-35B-A3B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.6 --size-in-billions 35 --model-format awq --quantization ${quantization}


Model Spec 4 (ggufv2, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 35
- **Quantizations:** MXFP4_MOE, Q8_0, UD-IQ1_M, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, UD-IQ4_NL_XL, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_M, UD-Q3_K_S, UD-Q3_K_XL, UD-Q4_K_M, UD-Q4_K_S, UD-Q4_K_XL, UD-Q5_K_M, UD-Q5_K_S, UD-Q5_K_XL, UD-Q6_K, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: 
- **Model ID:** unsloth/Qwen3.6-35B-A3B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.6-35B-A3B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.6 --size-in-billions 35 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (mlx, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 35
- **Quantizations:** 4bit, 5bit, 6bit, 8bit, bf16
- **Engines**: 
- **Model ID:** mlx-community/Qwen3.6-35B-A3B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.6-35B-A3B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.6 --size-in-billions 35 --model-format mlx --quantization ${quantization}

