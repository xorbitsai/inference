.. _models_llm_ornith-1.0-35b:

========================================
Ornith-1.0-35B
========================================

- **Context Length:** 262144
- **Model Name:** Ornith-1.0-35B
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Ornith-1.0-35B is a 35B-total / 3B-activated Mixture-of-Experts multimodal model built on the Qwen3.5 MoE architecture (Qwen3_5MoeForConditionalGeneration). It combines hybrid linear/full attention, 256 routed experts (8 per token) plus a shared expert, multimodal RoPE, and multi-token prediction, with vision and video understanding via the Qwen3VL processor.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepreinforce-ai/Ornith-1.0-35B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B>`__, `ModelScope <https://modelscope.cn/models/deepreinforce-ai/Ornith-1.0-35B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ornith-1.0-35B --size-in-billions 35 --model-format pytorch --quantization ${quantization}

