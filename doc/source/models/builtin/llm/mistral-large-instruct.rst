.. _models_llm_mistral-large-instruct:

========================================
mistral-large-instruct
========================================

- **Context Length:** 131072
- **Model Name:** mistral-large-instruct
- **Languages:** en, fr, de, es, it, pt, zh, ru, ja, ko
- **Abilities:** chat
- **Description:** Mistral-Large-Instruct-2407 is an advanced dense Large Language Model (LLM) of 123B parameters with state-of-the-art reasoning, knowledge and coding capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 123
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** mistralai/Mistral-Large-Instruct-2407
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mistral-Large-Instruct-2407>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Mistral-Large-Instruct-2407-bnb-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 123
- **Quantizations:** 4-bit
- **Engines**: Transformers
- **Model ID:** unsloth/Mistral-Large-Instruct-2407-bnb-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Mistral-Large-Instruct-2407-bnb-4bit>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Mistral-Large-Instruct-2407-bnb-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format pytorch --quantization ${quantization}


Model Spec 3 (gptq, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 123
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** ModelCloud/Mistral-Large-Instruct-2407-gptq-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/ModelCloud/Mistral-Large-Instruct-2407-gptq-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format gptq --quantization ${quantization}


Model Spec 4 (awq, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 123
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TechxGenus/Mistral-Large-Instruct-2407-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TechxGenus/Mistral-Large-Instruct-2407-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format awq --quantization ${quantization}


Model Spec 5 (ggufv2, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 123
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M
- **Engines**: llama.cpp
- **Model ID:** MaziyarPanahi/Mistral-Large-Instruct-2407-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MaziyarPanahi/Mistral-Large-Instruct-2407-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (mlx, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 123
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Mistral-Large-Instruct-2407-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Mistral-Large-Instruct-2407-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format mlx --quantization ${quantization}


Model Spec 7 (mlx, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 123
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Mistral-Large-Instruct-2407-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Mistral-Large-Instruct-2407-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format mlx --quantization ${quantization}


Model Spec 8 (mlx, 123 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 123
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Mistral-Large-Instruct-2407-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Mistral-Large-Instruct-2407-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-large-instruct --size-in-billions 123 --model-format mlx --quantization ${quantization}

