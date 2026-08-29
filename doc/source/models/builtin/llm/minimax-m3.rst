.. _models_llm_minimax-m3:

========================================
MiniMax-M3
========================================

- **Context Length:** 1048576
- **Model Name:** MiniMax-M3
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** MiniMax-M3 is a native multimodal model with 1M context. It has ~428B parameters and ~23B activated parameters.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 428 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 428
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** MiniMaxAI/MiniMax-M3
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MiniMaxAI/MiniMax-M3>`__, `ModelScope <https://modelscope.cn/models/MiniMax/MiniMax-M3>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M3 --size-in-billions 428 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 428 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 428
- **Quantizations:** none
- **Engines**: llama.cpp
- **Model ID:** unsloth/MiniMax-M3-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/MiniMax-M3-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/MiniMax-M3-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M3 --size-in-billions 428 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (mlx, 428 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 428
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/MiniMax-M3-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/MiniMax-M3-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/MiniMax-M3-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M3 --size-in-billions 428 --model-format mlx --quantization ${quantization}

