.. _models_llm_codestral-v0.1:

========================================
codestral-v0.1
========================================

- **Context Length:** 32768
- **Model Name:** codestral-v0.1
- **Languages:** en
- **Abilities:** generate
- **Description:** Codestrall-22B-v0.1 is trained on a diverse dataset of 80+ programming languages, including the most popular ones, such as Python, Java, C, C++, JavaScript, and Bash

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 22 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 22
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** mistralai/Mistral-7B-Instruct-v0.2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codestral-v0.1 --size-in-billions 22 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 22 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 22
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** bartowski/Codestral-22B-v0.1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codestral-v0.1 --size-in-billions 22 --model-format ggufv2 --quantization ${quantization}

