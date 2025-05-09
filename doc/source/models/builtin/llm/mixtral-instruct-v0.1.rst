.. _models_llm_mixtral-instruct-v0.1:

========================================
mixtral-instruct-v0.1
========================================

- **Context Length:** 32768
- **Model Name:** mixtral-instruct-v0.1
- **Languages:** en, fr, it, de, es
- **Abilities:** chat
- **Description:** Mistral-8x7B-Instruct is a fine-tuned version of the Mistral-8x7B LLM, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 46_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 46_7
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** mistralai/Mixtral-8x7B-Instruct-v0.1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Mixtral-8x7B-Instruct-v0.1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mixtral-instruct-v0.1 --size-in-billions 46_7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 46_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 46_7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mixtral-instruct-v0.1 --size-in-billions 46_7 --model-format awq --quantization ${quantization}


Model Spec 3 (gptq, 46_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 46_7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mixtral-instruct-v0.1 --size-in-billions 46_7 --model-format gptq --quantization ${quantization}


Model Spec 4 (ggufv2, 46_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 46_7
- **Quantizations:** Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q6_K, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mixtral-instruct-v0.1 --size-in-billions 46_7 --model-format ggufv2 --quantization ${quantization}

