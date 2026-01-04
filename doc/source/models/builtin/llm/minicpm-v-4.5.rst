.. _models_llm_minicpm-v-4.5:

========================================
MiniCPM-V-4.5
========================================

- **Context Length:** 32768
- **Model Name:** MiniCPM-V-4.5
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** MiniCPM-V 4.5 is an improved version in the MiniCPM-V series with enhanced multimodal capabilities and better performance.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** openbmb/MiniCPM-V-4_5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4_5>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.5 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** openbmb/MiniCPM-V-4_5-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4_5-int4>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4_5-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.5 --size-in-billions 8 --model-format pytorch --quantization ${quantization}

