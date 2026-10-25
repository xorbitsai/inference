.. _models_llm_vibethinker:

========================================
vibethinker
========================================

- **Context Length:** 131072
- **Model Name:** vibethinker
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** VibeThinker is a series of dense reasoning language models developed by WeiboAI. Built on the Qwen2 architecture with a post-training methodology centered on the Spectrum-to-Signal Principle (SSP), VibeThinker demonstrates strong reasoning capabilities in mathematics and coding despite its compact size.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** WeiboAI/VibeThinker-1.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WeiboAI/VibeThinker-1.5B>`__, `ModelScope <https://modelscope.cn/models/WeiboAI/VibeThinker-1.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name vibethinker --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** WeiboAI/VibeThinker-3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WeiboAI/VibeThinker-3B>`__, `ModelScope <https://modelscope.cn/models/WeiboAI/VibeThinker-3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name vibethinker --size-in-billions 3 --model-format pytorch --quantization ${quantization}

