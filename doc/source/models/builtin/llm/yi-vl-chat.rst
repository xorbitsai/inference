.. _models_llm_yi-vl-chat:

========================================
yi-vl-chat
========================================

- **Context Length:** 4096
- **Model Name:** yi-vl-chat
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** Yi Vision Language (Yi-VL) model is the open-source, multimodal version of the Yi Large Language Model (LLM) series, enabling content comprehension, recognition, and multi-round conversations about images.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** 01-ai/Yi-VL-6B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-VL-6B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-VL-6B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name yi-vl-chat --size-in-billions 6 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** 01-ai/Yi-VL-34B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-VL-34B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-VL-34B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name yi-vl-chat --size-in-billions 34 --model-format pytorch --quantization ${quantization}

