.. _models_llm_ornith-1.5-35b-a3b:

========================================
Ornith-1.5-35B-A3B
========================================

- **Context Length:** 262144
- **Model Name:** Ornith-1.5-35B-A3B
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Ornith-1.5-35B-A3B is a 35B-total / 3B-activated Mixture-of-Experts multimodal model built on the Qwen3.5 MoE architecture (Qwen3_5MoeForConditionalGeneration), supporting text, image, and video understanding with reasoning and tool-use capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** ornith-ai/Ornith-1.5-35B-A3B
- **Model Hubs**:  `ModelScope <https://modelscope.cn/models/ornith-ai/Ornith-1.5-35B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ornith-1.5-35B-A3B --size-in-billions 35 --model-format pytorch --quantization ${quantization}

