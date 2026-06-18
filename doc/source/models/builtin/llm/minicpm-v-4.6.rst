.. _models_llm_minicpm-v-4.6:

========================================
MiniCPM-V-4.6
========================================

- **Context Length:** 262144
- **Model Name:** MiniCPM-V-4.6
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** MiniCPM-V 4.6 is the latest and most edge-deployment-friendly model in the MiniCPM-V series, with only 1.3B parameters (1.3B activated). It is built on SigLIP2-400M and Qwen3.5-0.8B, and supports single-image, multi-image, and video understanding.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: Transformers, SGLang
- **Model ID:** openbmb/MiniCPM-V-4.6
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.6>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.6 --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (bnb, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** bnb
- **Model Size (in billions):** 1
- **Quantizations:** 4-bit
- **Engines**: Transformers, SGLang
- **Model ID:** openbmb/MiniCPM-V-4.6-BNB
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.6-BNB>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-BNB>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.6 --size-in-billions 1 --model-format bnb --quantization ${quantization}


Model Spec 3 (awq, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1
- **Quantizations:** Int4
- **Engines**: Transformers, SGLang
- **Model ID:** openbmb/MiniCPM-V-4.6-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.6-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.6 --size-in-billions 1 --model-format awq --quantization ${quantization}


Model Spec 4 (gptq, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1
- **Quantizations:** Int4
- **Engines**: Transformers, SGLang
- **Model ID:** openbmb/MiniCPM-V-4.6-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.6 --size-in-billions 1 --model-format gptq --quantization ${quantization}


Model Spec 5 (ggufv2, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1
- **Quantizations:** Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, F16
- **Engines**: llama.cpp
- **Model ID:** openbmb/MiniCPM-V-4.6-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.6 --size-in-billions 1 --model-format ggufv2 --quantization ${quantization}

