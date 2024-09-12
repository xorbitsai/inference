.. _models_llm_minicpm3-4b:

========================================
minicpm3-4b
========================================

- **Context Length:** 32768
- **Model Name:** minicpm3-4b
- **Languages:** zh
- **Abilities:** chat
- **Description:** MiniCPM3-4B is the 3rd generation of MiniCPM series. The overall performance of MiniCPM3-4B surpasses Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125, being comparable with many recent 7B~9B models.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM3-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM3-4B>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM3-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm3-4b --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM3-4B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM3-4B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM3-4B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm3-4b --size-in-billions 4 --model-format gptq --quantization ${quantization}

