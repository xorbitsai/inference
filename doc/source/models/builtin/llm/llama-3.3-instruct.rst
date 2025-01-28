.. _models_llm_llama-3.3-instruct:

========================================
llama-3.3-instruct
========================================

- **Context Length:** 131072
- **Model Name:** llama-3.3-instruct
- **Languages:** en, de, fr, it, pt, hi, es, th
- **Abilities:** chat, tools
- **Description:** The Llama 3.3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks..

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Llama-3.3-70B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Llama-3.3-70B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.3-instruct --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** shuyuej/Llama-3.3-70B-Instruct-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/shuyuej/Llama-3.3-70B-Instruct-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.3-instruct --size-in-billions 70 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** casperhansen/llama-3.3-70b-instruct-awq
- **Model Hubs**:  `Hugging Face <https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.3-instruct --size-in-billions 70 --model-format awq --quantization ${quantization}


Model Spec 4 (mlx, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 70
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, fp16
- **Engines**: MLX
- **Model ID:** mlx-community/Llama-3.3-70B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.3-instruct --size-in-billions 70 --model-format mlx --quantization ${quantization}


Model Spec 5 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** Q3_K_L, Q4_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Llama-3.3-70B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Llama-3.3-70B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/lmstudio-community/Llama-3.3-70B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.3-instruct --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}

