.. _models_llm_mistral-instruct-v0.1:

========================================
mistral-instruct-v0.1
========================================

- **Context Length:** 8192
- **Model Name:** mistral-instruct-v0.1
- **Languages:** en
- **Abilities:** chat
- **Description:** Mistral-7B-Instruct is a fine-tuned version of the Mistral-7B LLM on public datasets, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** mistralai/Mistral-7B-Instruct-v0.1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1>`__, `ModelScope <https://modelscope.cn/models/Xorbits/Mistral-7B-Instruct-v0.1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.1 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/Mistral-7B-Instruct-v0.1-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.1 --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 3 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/Mistral-7B-Instruct-v0.1-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.1 --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 4 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/Mistral-7B-Instruct-v0.1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.1 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

