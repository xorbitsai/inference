.. _models_llm_ornith-1.5-397b:

========================================
Ornith-1.5-397B
========================================

- **Context Length:** 262144
- **Model Name:** Ornith-1.5-397B
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Ornith-1.5-397B is a 397B-total Mixture-of-Experts multimodal model built on the Qwen3.5 MoE architecture (Qwen3_5MoeForConditionalGeneration), supporting text, image, and video understanding with reasoning and tool-use capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 397
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** ornith-ai/Ornith-1.5-397B
- **Model Hubs**:  `ModelScope <https://modelscope.cn/models/ornith-ai/Ornith-1.5-397B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ornith-1.5-397B --size-in-billions 397 --model-format pytorch --quantization ${quantization}

