.. _models_llm_glm-5.2:

========================================
glm-5.2
========================================

- **Context Length:** 1048576
- **Model Name:** glm-5.2
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning, hybrid
- **Description:** We're introducing GLM-5.2, our latest flagship model for long-horizon tasks

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 753 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 753
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** zai-org/GLM-5.2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-5.2>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-5.2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.2 --size-in-billions 753 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 753 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 753
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** zai-org/GLM-5.2-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-5.2-FP8>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-5.2-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.2 --size-in-billions 753 --model-format fp8 --quantization ${quantization}


Model Spec 3 (ggufv2, 753 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 753
- **Quantizations:** BF16, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_M, UD-Q3_K_XL, UD-Q4_K_M, UD-Q4_K_S, UD-Q4_K_XL, UD-Q5_K_M, UD-Q5_K_S, UD-Q5_K_XL, UD-Q6_K, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/GLM-5.2-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/GLM-5.2-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/GLM-5.2-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.2 --size-in-billions 753 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (mlx, 753 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 753
- **Quantizations:** 4bit, mxfp4
- **Engines**: MLX
- **Model ID:** mlx-community/GLM-5.2-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/GLM-5.2-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/GLM-5.2-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.2 --size-in-billions 753 --model-format mlx --quantization ${quantization}

