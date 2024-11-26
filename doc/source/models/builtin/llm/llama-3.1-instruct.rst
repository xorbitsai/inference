.. _models_llm_llama-3.1-instruct:

========================================
llama-3.1-instruct
========================================

- **Context Length:** 131072
- **Model Name:** llama-3.1-instruct
- **Languages:** en, de, fr, it, pt, hi, es, th
- **Abilities:** chat, tools
- **Description:** The Llama 3.1 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks..

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 8
- **Quantizations:** Q3_K_L, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Meta-Llama-3.1-8B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit
- **Engines**: Transformers
- **Model ID:** unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 4 (gptq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format gptq --quantization ${quantization}


Model Spec 5 (awq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format awq --quantization ${quantization}


Model Spec 6 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** IQ2_M, IQ4_XS, Q2_K, Q3_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Meta-Llama-3.1-70B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 8 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit
- **Engines**: Transformers
- **Model ID:** unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 9 (gptq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format gptq --quantization ${quantization}


Model Spec 10 (awq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format awq --quantization ${quantization}


Model Spec 11 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 12 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-Llama-3.1-8B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 13 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-Llama-3.1-8B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 14 (mlx, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-Llama-3.1-70B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-Llama-3.1-70B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format mlx --quantization ${quantization}


Model Spec 15 (mlx, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 70
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Meta-Llama-3.1-70B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Meta-Llama-3.1-70B-Instruct-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 70 --model-format mlx --quantization ${quantization}


Model Spec 16 (mlx, 70 Billion)
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


Model Spec 17 (pytorch, 405 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 405
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** meta-llama/Meta-Llama-3.1-405B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-405B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 405 --model-format pytorch --quantization ${quantization}


Model Spec 18 (gptq, 405 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 405
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 405 --model-format gptq --quantization ${quantization}


Model Spec 19 (awq, 405 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 405
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-405B-Instruct-AWQ-INT4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.1-instruct --size-in-billions 405 --model-format awq --quantization ${quantization}

