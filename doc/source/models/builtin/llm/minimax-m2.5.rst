.. _models_llm_minimax-m2.5:

========================================
MiniMax-M2.5
========================================

- **Context Length:** 196608
- **Model Name:** MiniMax-M2.5
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning
- **Description:** MiniMax-M2.5, a Mini model built for Max coding & agentic workflows.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 230 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 230
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** MiniMaxAI/MiniMax-M2.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MiniMaxAI/MiniMax-M2.5>`__, `ModelScope <https://modelscope.cn/models/MiniMax/MiniMax-M2.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M2.5 --size-in-billions 230 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 230 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 230
- **Quantizations:** UD-TQ1_0
- **Engines**: 
- **Model ID:** unsloth/MiniMax-M2.5-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/MiniMax-M2.5-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/MiniMax-M2.5-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M2.5 --size-in-billions 230 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (mlx, 230 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 230
- **Quantizations:** 3bit, 4bit, 5bit, 6bit, 8bit
- **Engines**: 
- **Model ID:** mlx-community/MiniMax-M2.5-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/MiniMax-M2.5-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/MiniMax-M2.5-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M2.5 --size-in-billions 230 --model-format mlx --quantization ${quantization}

