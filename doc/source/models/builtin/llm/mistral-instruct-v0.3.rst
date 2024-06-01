.. _models_llm_mistral-instruct-v0.3:

========================================
mistral-instruct-v0.3
========================================

- **Context Length:** 32768
- **Model Name:** mistral-instruct-v0.3
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
- **Engines**: Transformers
- **Model ID:** mistralai/Mistral-7B-Instruct-v0.3
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.3 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.3 --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** solidrust/Mistral-7B-Instruct-v0.3-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/solidrust/Mistral-7B-Instruct-v0.3-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.3 --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 4 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-instruct-v0.3 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

