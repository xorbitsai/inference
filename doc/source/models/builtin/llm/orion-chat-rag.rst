.. _models_llm_orion-chat-rag:

========================================
orion-chat-rag
========================================

- **Context Length:** 4096
- **Model Name:** orion-chat-rag
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Orion-14B series models are open-source multilingual large language models trained from scratch by OrionStarAI.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none, 4-bit, 8-bit
- **Engines**: vLLM, Transformers (vllm only available for quantization none)
- **Model ID:** OrionStarAI/Orion-14B-Chat-RAG
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OrionStarAI/Orion-14B-Chat-RAG>`__, `ModelScope <https://modelscope.cn/models/OrionStarAI/Orion-14B-Chat-RAG>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name orion-chat-rag --size-in-billions 14 --model-format pytorch --quantization ${quantization}

