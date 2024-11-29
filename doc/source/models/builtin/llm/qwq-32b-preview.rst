.. _models_llm_qwq-32b-preview:

========================================
QwQ-32B-Preview
========================================

- **Context Length:** 32768
- **Model Name:** QwQ-32B-Preview
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** QwQ-32B-Preview is an experimental research model developed by the Qwen Team, focused on advancing AI reasoning capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/QwQ-32B-Preview
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/QwQ-32B-Preview>`__, `ModelScope <https://modelscope.cn/models/Qwen/QwQ-32B-Preview>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QwQ-32B-Preview --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** KirillR/QwQ-32B-Preview-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/KirillR/QwQ-32B-Preview-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QwQ-32B-Preview --size-in-billions 32 --model-format awq --quantization ${quantization}


Model Spec 3 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** Q3_K_L, Q4_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/QwQ-32B-Preview-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/QwQ-32B-Preview-GGUF>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/QwQ-32B-Preview-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QwQ-32B-Preview --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen_QwQ-32B-Preview_MLX-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen_QwQ-32B-Preview_MLX-4bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/QwQ-32B-Preview-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QwQ-32B-Preview --size-in-billions 32 --model-format mlx --quantization ${quantization}


Model Spec 5 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen_QwQ-32B-Preview_MLX-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen_QwQ-32B-Preview_MLX-8bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/QwQ-32B-Preview-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QwQ-32B-Preview --size-in-billions 32 --model-format mlx --quantization ${quantization}


Model Spec 6 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/QwQ-32B-Preview-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/QwQ-32B-Preview-bf16>`__, `ModelScope <https://modelscope.cn/models/okwinds/QwQ-32B-Preview-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QwQ-32B-Preview --size-in-billions 32 --model-format mlx --quantization ${quantization}

