.. _models_llm_starling-lm:

========================================
Starling-LM
========================================

- **Context Length:** 4096
- **Model Name:** Starling-LM
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** We introduce Starling-7B, an open large language model (LLM) trained by Reinforcement Learning from AI Feedback (RLAIF). The model harnesses the power of our new GPT-4 labeled ranking dataset

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** berkeley-nest/Starling-LM-7B-alpha
- **Model Hubs**:  `Hugging Face <https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Starling-LM --size-in-billions 7 --model-format pytorch --quantization ${quantization}

