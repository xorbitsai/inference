.. _models_llm_mixtral-8x22b-instruct-v0.1:

========================================
mixtral-8x22B-instruct-v0.1
========================================

- **Context Length:** 65536
- **Model Name:** mixtral-8x22B-instruct-v0.1
- **Languages:** en, fr, it, de, es
- **Abilities:** chat
- **Description:** The Mixtral-8x22B-Instruct-v0.1 Large Language Model (LLM) is an instruct fine-tuned version of the Mixtral-8x22B-v0.1, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 141 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 141
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** mistralai/Mixtral-8x22B-Instruct-v0.1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mixtral-8x22B-instruct-v0.1 --size-in-billions 141 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 141 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 141
- **Quantizations:** Int4
- **Model ID:** MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mixtral-8x22B-instruct-v0.1 --size-in-billions 141 --model-format awq --quantization ${quantization}


Model Spec 3 (gptq, 141 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 141
- **Quantizations:** Int4
- **Model ID:** jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/jarrelscy/Mixtral-8x22B-Instruct-v0.1-GPTQ-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mixtral-8x22B-instruct-v0.1 --size-in-billions 141 --model-format gptq --quantization ${quantization}


Model Spec 4 (ggufv2, 141 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 141
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6, Q8_0, fp16
- **Model ID:** MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mixtral-8x22B-instruct-v0.1 --size-in-billions 141 --model-format ggufv2 --quantization ${quantization}

