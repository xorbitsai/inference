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
- **Engines**: Transformers
- **Model ID:** MiniMaxAI/MiniMax-M2.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MiniMaxAI/MiniMax-M2.5>`__, `ModelScope <https://modelscope.cn/models/MiniMax/MiniMax-M2.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M2.5 --size-in-billions 230 --model-format pytorch --quantization ${quantization}

