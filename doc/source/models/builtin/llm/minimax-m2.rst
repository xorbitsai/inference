.. _models_llm_minimax-m2:

========================================
MiniMax-M2
========================================

- **Context Length:** 196608
- **Model Name:** MiniMax-M2
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning
- **Description:** MiniMax-M2, a Mini model built for Max coding & agentic workflows.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 230 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 230
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** MiniMaxAI/MiniMax-M2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MiniMaxAI/MiniMax-M2>`__, `ModelScope <https://modelscope.cn/models/MiniMax/MiniMax-M2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M2 --size-in-billions 230 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 230 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 230
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** QuantTrio/MiniMax-M2-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/MiniMax-M2-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/MiniMax-M2-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M2 --size-in-billions 230 --model-format awq --quantization ${quantization}


Model Spec 3 (mlx, 230 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 230
- **Quantizations:** 3bit, 4bit, 5bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/MiniMax-M2-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/MiniMax-M2-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/MiniMax-M2-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M2 --size-in-billions 230 --model-format mlx --quantization ${quantization}

