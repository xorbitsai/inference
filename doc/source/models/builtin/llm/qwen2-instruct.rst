.. _models_llm_qwen2-instruct:

========================================
qwen2-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2-instruct
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Qwen2 is the new series of Qwen large language models

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2-0.5B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 0_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2-1.5B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-1.5B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2-72B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-72B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-72B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 5 (gptq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-0.5B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 0_5 --model-format gptq --quantization ${quantization}


Model Spec 6 (gptq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-1.5B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 1_5 --model-format gptq --quantization ${quantization}


Model Spec 7 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-7B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-7B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 8 (gptq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 72
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-72B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-72B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-72B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 72 --model-format gptq --quantization ${quantization}


Model Spec 9 (awq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-0.5B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 0_5 --model-format awq --quantization ${quantization}


Model Spec 10 (awq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-1.5B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 1_5 --model-format awq --quantization ${quantization}


Model Spec 11 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-7B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-7B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-7B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 12 (awq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 72
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2-72B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-72B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-72B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 72 --model-format awq --quantization ${quantization}


Model Spec 13 (mlx, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** Qwen/Qwen2-0.5B-Instruct-MLX
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-MLX>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct-MLX>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 0_5 --model-format mlx --quantization ${quantization}


Model Spec 14 (mlx, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** Qwen/Qwen2-1.5B-Instruct-MLX
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-MLX>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct-MLX>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 1_5 --model-format mlx --quantization ${quantization}


Model Spec 15 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** Qwen/Qwen2-7B-Instruct-MLX
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-7B-Instruct-MLX>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-7B-Instruct-MLX>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 7 --model-format mlx --quantization ${quantization}


Model Spec 16 (mlx, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 72
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2-72B-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2-72B-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 72 --model-format mlx --quantization ${quantization}


Model Spec 17 (ggufv2, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 0_5
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2-0.5B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 0_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 18 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2-1.5B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 19 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2-7B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-7B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 20 (ggufv2, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 72
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2-72B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-72B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-72B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-instruct --size-in-billions 72 --model-format ggufv2 --quantization ${quantization}

