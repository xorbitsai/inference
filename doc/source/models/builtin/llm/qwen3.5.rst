.. _models_llm_qwen3.5:

========================================
qwen3.5
========================================

- **Context Length:** 262144
- **Model Name:** qwen3.5
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning
- **Description:** Over recent months, we have intensified our focus on developing foundation models that deliver exceptional utility and performance. Qwen3.5 represents a significant leap forward, integrating breakthroughs in multimodal learning, architectural efficiency, reinforcement learning scale, and global accessibility to empower developers and enterprises with unprecedented capability and efficiency.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 397
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-397B-A17B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format pytorch --quantization ${quantization}

