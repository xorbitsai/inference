.. _models_llm_minicpm-v-4.7:

========================================
MiniCPM-V-4.7
========================================

- **Context Length:** 262144
- **Model Name:** MiniCPM-V-4.7
- **Languages:** en, zh
- **Abilities:** chat, vision, hybrid
- **Description:** MiniCPM-V 4.7 supports single-image, multi-image, and video understanding with canvas M-RoPE and 4x/16x visual compression. It provides 1B dense and 35B-A3B MoE variants. Transformers requires >=5.18.0; vLLM requires the nightly containing PR #58674. SGLang support currently requires an upstream fork with conflicting dependency pins and is not enabled. GGUF and llama.cpp support is not enabled.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-1B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-1B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 1 --model-format awq --quantization ${quantization}


Model Spec 3 (gptq, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-1B-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-GPTQ>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 1 --model-format gptq --quantization ${quantization}


Model Spec 4 (bnb, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** bnb
- **Model Size (in billions):** 1
- **Quantizations:** 4-bit
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-1B-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-int4>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 1 --model-format bnb --quantization ${quantization}


Model Spec 5 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-35B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 6 (awq, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 35
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-35B-A3B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 35 --model-format awq --quantization ${quantization}


Model Spec 7 (gptq, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 35
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-35B-A3B-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-GPTQ>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 35 --model-format gptq --quantization ${quantization}


Model Spec 8 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-35B-A3B-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-int4>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 9 (fp8, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 35
- **Quantizations:** FP8
- **Engines**: vLLM, Transformers
- **Model ID:** openbmb/MiniCPM-V-4.7-35B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 35 --model-format fp8 --quantization ${quantization}
