.. _models_llm_minicpm-v-4.7:

========================================
MiniCPM-V-4.7
========================================

- **Context Length:** 262144
- **Model Name:** MiniCPM-V-4.7
- **Languages:** en, zh
- **Abilities:** chat, vision, hybrid
- **Description:** MiniCPM-V 4.7 supports single-image, multi-image, and video understanding with canvas M-RoPE and 4x/16x visual compression. It provides 1B dense and 35B-A3B MoE variants.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
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
- **Engines**: vLLM, Transformers, SGLang
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
- **Engines**: vLLM, Transformers, SGLang
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
- **Engines**: vLLM, Transformers, SGLang
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
- **Engines**: vLLM, Transformers, SGLang
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
- **Engines**: vLLM, Transformers, SGLang
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
- **Engines**: vLLM, Transformers, SGLang
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
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** openbmb/MiniCPM-V-4.7-35B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 35 --model-format fp8 --quantization ${quantization}


Model Spec 10 (ggufv2, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1
- **Quantizations:** Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, F16, IQ3_M, IQ3_XS, IQ4_NL, IQ4_XS, Q4_0, Q4_1, Q5_0, Q5_1
- **Engines**: llama.cpp
- **Model ID:** openbmb/MiniCPM-V-4.7-1B-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-1B-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 1 --model-format ggufv2 --quantization ${quantization}


Model Spec 11 (ggufv2, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 35
- **Quantizations:** Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, F16, IQ3_M, IQ3_XS, IQ4_NL, IQ4_XS, Q4_0, Q4_1, Q5_0, Q5_1, mix_q3gud_im
- **Engines**: llama.cpp
- **Model ID:** openbmb/MiniCPM-V-4.7-35B-A3B-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.7-35B-A3B-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniCPM-V-4.7 --size-in-billions 35 --model-format ggufv2 --quantization ${quantization}
