.. _models_llm_qwen3.8:

========================================
qwen3.8
========================================

- **Context Length:** 262144
- **Model Name:** qwen3.8
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 27
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3.8-27B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.8-27B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.8-27B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8 --size-in-billions 27 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 27
- **Quantizations:** FP8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3.8-27B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.8-27B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8 --size-in-billions 27 --model-format fp8 --quantization ${quantization}


Model Spec 3 (ggufv2, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 27
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: llama.cpp
- **Model ID:** unsloth/Qwen3.8-27B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8 --size-in-billions 27 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (mlx, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 27
- **Quantizations:** 4bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3.8-27B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3.8-27B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3.8-27B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8 --size-in-billions 27 --model-format mlx --quantization ${quantization}
