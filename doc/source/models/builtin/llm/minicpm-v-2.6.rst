.. _models_llm_minicpm-v-2.6:

========================================
MiniCPM-V-2.6
========================================

- **Context Length:** 32768
- **Model Name:** MiniCPM-V-2.6
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** MiniCPM-V 2.6 is the latest model in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B with a total of 8B parameters.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM-V-2_6
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-2_6>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-2.6 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM-V-2_6-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-2_6-int4>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-2.6 --size-in-billions 8 --model-format pytorch --quantization ${quantization}

