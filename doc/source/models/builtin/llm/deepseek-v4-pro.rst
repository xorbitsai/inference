.. _models_llm_deepseek-v4-pro:

========================================
DeepSeek-V4-Pro
========================================

- **Context Length:** 163840
- **Model Name:** DeepSeek-V4-Pro
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of one million tokens.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1600 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1600
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/DeepSeek-V4-Pro
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Pro>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4-Pro --size-in-billions 1600 --model-format pytorch --quantization ${quantization}


Model Spec 2 (mlx, 1600 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1600
- **Quantizations:** 4bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-V4-Pro-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-V4-Pro-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/DeepSeek-V4-Pro-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4-Pro --size-in-billions 1600 --model-format mlx --quantization ${quantization}

