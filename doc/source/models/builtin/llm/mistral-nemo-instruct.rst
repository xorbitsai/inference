.. _models_llm_mistral-nemo-instruct:

========================================
mistral-nemo-instruct
========================================

- **Context Length:** 1024000
- **Model Name:** mistral-nemo-instruct
- **Languages:** en, fr, de, es, it, pt, zh, ru, ja
- **Abilities:** chat
- **Description:** The Mistral-Nemo-Instruct-2407 Large Language Model (LLM) is an instruct fine-tuned version of the Mistral-Nemo-Base-2407

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 12
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** mistralai/Mistral-Nemo-Instruct-2407
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Mistral-Nemo-Instruct-2407>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 12
- **Quantizations:** 4-bit
- **Engines**: Transformers
- **Model ID:** unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Mistral-Nemo-Instruct-2407>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 12
- **Quantizations:** 8-bit
- **Engines**: Transformers
- **Model ID:** afrizalha/Mistral-Nemo-Instruct-2407-bnb-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/afrizalha/Mistral-Nemo-Instruct-2407-bnb-8bit>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Mistral-Nemo-Instruct-2407>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format pytorch --quantization ${quantization}


Model Spec 4 (gptq, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 12
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** ModelCloud/Mistral-Nemo-Instruct-2407-gptq-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/ModelCloud/Mistral-Nemo-Instruct-2407-gptq-4bit>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Mistral-Nemo-Instruct-2407-gptq-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format gptq --quantization ${quantization}


Model Spec 5 (awq, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 12
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** casperhansen/mistral-nemo-instruct-2407-awq
- **Model Hubs**:  `Hugging Face <https://huggingface.co/casperhansen/mistral-nemo-instruct-2407-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format awq --quantization ${quantization}


Model Spec 6 (ggufv2, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 12
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (mlx, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 12
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Mistral-Nemo-Instruct-2407-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Mistral-Nemo-Instruct-2407-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format mlx --quantization ${quantization}


Model Spec 8 (mlx, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 12
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Mistral-Nemo-Instruct-2407-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Mistral-Nemo-Instruct-2407-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format mlx --quantization ${quantization}


Model Spec 9 (mlx, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 12
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Mistral-Nemo-Instruct-2407-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Mistral-Nemo-Instruct-2407-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name mistral-nemo-instruct --size-in-billions 12 --model-format mlx --quantization ${quantization}

