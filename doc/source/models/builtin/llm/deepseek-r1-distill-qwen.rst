.. _models_llm_deepseek-r1-distill-qwen:

========================================
deepseek-r1-distill-qwen
========================================

- **Context Length:** 131072
- **Model Name:** deepseek-r1-distill-qwen
- **Languages:** en, zh
- **Abilities:** chat, reasoning
- **Description:** deepseek-r1-distill-qwen is distilled from DeepSeek-R1 based on Qwen

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** casperhansen/deepseek-r1-distill-qwen-1.5b-awq
- **Model Hubs**:  `Hugging Face <https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-1.5b-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 1_5 --model-format awq --quantization ${quantization}


Model Spec 3 (gptq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** jakiAJK/DeepSeek-R1-Distill-Qwen-1.5B_GPTQ-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/jakiAJK/DeepSeek-R1-Distill-Qwen-1.5B_GPTQ-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 1_5 --model-format gptq --quantization ${quantization}


Model Spec 4 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (mlx, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1_5
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 1_5 --model-format mlx --quantization ${quantization}


Model Spec 6 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 7 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 8 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** jakiAJK/DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/jakiAJK/DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4>`__, `ModelScope <https://modelscope.cn/models/tclf90/deepseek-r1-distill-qwen-7b-gptq-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 9 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 10 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-R1-Distill-Qwen-7B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-7B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/okwinds/DeepSeek-R1-Distill-Qwen-7B-MLX-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 7 --model-format mlx --quantization ${quantization}


Model Spec 11 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 12 (awq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** casperhansen/deepseek-r1-distill-qwen-14b-awq
- **Model Hubs**:  `Hugging Face <https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-14b-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 14 --model-format awq --quantization ${quantization}


Model Spec 13 (ggufv2, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 14
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 14 --model-format ggufv2 --quantization ${quantization}


Model Spec 14 (mlx, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 14
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-R1-Distill-Qwen-14B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-14B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/okwinds/DeepSeek-R1-Distill-Qwen-14B-MLX-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 14 --model-format mlx --quantization ${quantization}


Model Spec 15 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 16 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** casperhansen/deepseek-r1-distill-qwen-32b-awq
- **Model Hubs**:  `Hugging Face <https://huggingface.co/casperhansen/deepseek-r1-distill-qwen-32b-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 32 --model-format awq --quantization ${quantization}


Model Spec 17 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 18 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-R1-Distill-Qwen-32B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-32B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/okwinds/DeepSeek-R1-Distill-Qwen-32B-MLX-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-qwen --size-in-billions 32 --model-format mlx --quantization ${quantization}

