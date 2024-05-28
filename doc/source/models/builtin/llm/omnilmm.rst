.. _models_llm_omnilmm:

========================================
OmniLMM
========================================

- **Context Length:** 2048
- **Model Name:** OmniLMM
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** OmniLMM is a family of open-source large multimodal models (LMMs) adept at vision & language modeling.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM-V
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name OmniLMM --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 12
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/OmniLMM-12B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/OmniLMM-12B>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/OmniLMM-12B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name OmniLMM --size-in-billions 12 --model-format pytorch --quantization ${quantization}

