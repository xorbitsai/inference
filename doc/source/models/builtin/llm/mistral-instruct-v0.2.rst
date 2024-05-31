.. _models_llm_mistral-instruct-v0.2:

========================================
mistral-instruct-v0.2
========================================

- **Context Length:** 8192
- **Model Name:** mistral-instruct-v0.2
- **Languages:** en
- **Abilities:** chat
- **Description:** The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of Mistral-7B-Instruct-v0.1.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** mistralai/Mistral-7B-Instruct-v0.2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.2 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.2 --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/Mistral-7B-Instruct-v0.2-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.2 --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 4 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/Mistral-7B-Instruct-v0.2-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF>`__, `ModelScope <https://modelscope.cn/models/Xorbits/Mistral-7B-Instruct-v0.2-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.2 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

