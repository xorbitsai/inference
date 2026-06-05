.. _models_llm_deepseek-v4-flash:

========================================
DeepSeek-V4-Flash
========================================

- **Context Length:** 163840
- **Model Name:** DeepSeek-V4-Flash
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** We present a preview version of DeepSeek-V4 series, including two strong Mixture-of-Experts (MoE) language models — DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) — both supporting a context length of one million tokens.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 284 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 284
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** deepseek-ai/DeepSeek-V4-Flash
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4-Flash --size-in-billions 284 --model-format pytorch --quantization ${quantization}


Model Spec 2 (mlx, 284 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 284
- **Quantizations:** 4bit, 5bit, 6bit, 8bit, bf16, mxfp4, mxfp8, nvfp4
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-V4-Flash-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-V4-Flash-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/DeepSeek-V4-Flash-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V4-Flash --size-in-billions 284 --model-format mlx --quantization ${quantization}

