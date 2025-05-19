.. _models_llm_dianjin-r1:

========================================
DianJin-R1
========================================

- **Context Length:** 32768
- **Model Name:** DianJin-R1
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Tongyi DianJin is a financial intelligence solution platform built by Alibaba Cloud, dedicated to providing financial business developers with a convenient artificial intelligence application development environment.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** DianJin/DianJin-R1-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/DianJin/DianJin-R1-7B>`__, `ModelScope <https://modelscope.cn/models/DianJin/DianJin-R1-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DianJin-R1 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** DianJin/DianJin-R1-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/DianJin/DianJin-R1-32B>`__, `ModelScope <https://modelscope.cn/models/DianJin/DianJin-R1-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DianJin-R1 --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, f16
- **Engines**: llama.cpp
- **Model ID:** mradermacher/DianJin-R1-7B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mradermacher/DianJin-R1-7B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DianJin-R1 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** i1-IQ1_S, i1-IQ1_M, i1-IQ2_XXS, i1-IQ2_XS, i1-IQ2_S, i1-IQ2_M, i1-Q2_K_S, i1-Q2_K, i1-IQ3_XXS, i1-IQ3_XS, i1-Q3_K_S, i1-IQ3_S, i1-IQ3_M, i1-Q3_K_M, i1-Q3_K_L, i1-IQ4_XS, i1-IQ4_NL, i1-Q4_0, i1-Q4_K_S, i1-Q4_K_M, i1-Q4_1, i1-Q5_K_S, i1-Q5_K_M, i1-Q6_K
- **Engines**: llama.cpp
- **Model ID:** mradermacher/DianJin-R1-7B-i1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mradermacher/DianJin-R1-7B-i1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DianJin-R1 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** mradermacher/DianJin-R1-32B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mradermacher/DianJin-R1-32B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DianJin-R1 --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** i1-IQ1_S, i1-IQ1_M, i1-IQ2_XXS, i1-IQ2_XS, i1-IQ2_S, i1-IQ2_M, i1-Q2_K_S, i1-Q2_K, i1-IQ3_XXS, i1-IQ3_XS, i1-Q3_K_S, i1-IQ3_S, i1-IQ3_M, i1-Q3_K_M, i1-Q3_K_L, i1-IQ4_XS, i1-Q4_0, i1-Q4_K_S, i1-Q4_K_M, i1-Q4_1, i1-Q5_K_S, i1-Q5_K_M, i1-Q6_K
- **Engines**: llama.cpp
- **Model ID:** mradermacher/DianJin-R1-32B-i1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mradermacher/DianJin-R1-32B-i1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DianJin-R1 --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}

