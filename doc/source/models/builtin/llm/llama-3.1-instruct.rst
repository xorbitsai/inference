.. _models_llm_llama-3.1-instruct:

========================================
llama-3.1-instruct
========================================

- **Context Length:** 8192
- **Model Name:** llama-3.1-instruct
- **Languages:** en
- **Abilities:** chat
- **Description:** The Llama 3.1 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks..

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 8
- **Quantizations:** Q3_K_L, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Meta-llama-3.1-8B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Meta-llama-3.1-8B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** meta-llama/Meta-llama-3.1-8B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-llama-3.1-8B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-llama-3.1-8B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** IQ2_M, IQ4_XS, Q2_K, Q3_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Meta-llama-3.1-70B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Meta-llama-3.1-70B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** meta-llama/Meta-llama-3.1-70B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-llama-3.1-70B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-llama-3.1-70B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 5 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-llama-3.1-8B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-llama-3.1-8B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 6 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-llama-3.1-8B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-llama-3.1-8B-Instruct-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 7 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-llama-3.1-8B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-llama-3.1-8B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 8 (mlx, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-llama-3.1-70B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-llama-3.1-70B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format mlx --quantization ${quantization}


Model Spec 9 (mlx, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 70
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-llama-3.1-70B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-llama-3.1-70B-Instruct-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format mlx --quantization ${quantization}


Model Spec 10 (mlx, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-Llama-3.1-70B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-Llama-3.1-70B-Instruct-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format mlx --quantization ${quantization}

