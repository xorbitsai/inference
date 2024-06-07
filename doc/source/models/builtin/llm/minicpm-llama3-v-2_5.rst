.. _models_llm_minicpm-llama3-v-2_5:

========================================
MiniCPM-Llama3-V-2_5
========================================

- **Context Length:** 2048
- **Model Name:** MiniCPM-Llama3-V-2_5
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** MiniCPM-Llama3-V 2.5 is the latest model in the MiniCPM-V series. The model is built on SigLip-400M and Llama3-8B-Instruct with a total of 8B parameters.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM-Llama3-V-2_5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-Llama3-V-2_5 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** int4
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM-Llama3-V-2_5-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-{quantization}>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-Llama3-V-2_5 --size-in-billions 8 --model-format pytorch --quantization ${quantization}

