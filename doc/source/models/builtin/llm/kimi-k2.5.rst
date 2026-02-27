.. _models_llm_kimi-k2.5:

========================================
Kimi-K2.5
========================================

- **Context Length:** 262144
- **Model Name:** Kimi-K2.5
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** Kimi K2.5 is an open-source, native multimodal agentic model built through continual pretraining on approximately 15 trillion mixed visual and text tokens atop Kimi-K2-Base. It seamlessly integrates vision and language understanding with advanced agentic capabilities, instant and thinking modes, as well as conversational and agentic paradigms.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1058_59 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1058_59
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** moonshotai/Kimi-K2.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/moonshotai/Kimi-K2.5>`__, `ModelScope <https://modelscope.cn/models/moonshotai/Kimi-K2.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Kimi-K2.5 --size-in-billions 1058_59 --model-format pytorch --quantization ${quantization}

