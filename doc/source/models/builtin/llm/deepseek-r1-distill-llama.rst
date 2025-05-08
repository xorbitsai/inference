.. _models_llm_deepseek-r1-distill-llama:

========================================
deepseek-r1-distill-llama
========================================

- **Context Length:** 131072
- **Model Name:** deepseek-r1-distill-llama
- **Languages:** en, zh
- **Abilities:** chat, reasoning
- **Description:** deepseek-r1-distill-llama is distilled from DeepSeek-R1 based on Llama

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** jakiAJK/DeepSeek-R1-Distill-Llama-8B_AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/jakiAJK/DeepSeek-R1-Distill-Llama-8B_AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 8 --model-format awq --quantization ${quantization}


Model Spec 3 (gptq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 8 --model-format gptq --quantization ${quantization}


Model Spec 4 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-R1-Distill-Llama-8B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Llama-8B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/okwinds/DeepSeek-R1-Distill-Llama-8B-MLX-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 6 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 7 (awq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** casperhansen/deepseek-r1-distill-llama-70b-awq
- **Model Hubs**:  `Hugging Face <https://huggingface.co/casperhansen/deepseek-r1-distill-llama-70b-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 70 --model-format awq --quantization ${quantization}


Model Spec 8 (gptq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** empirischtech/DeepSeek-R1-Distill-Llama-70B-gptq-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/empirischtech/DeepSeek-R1-Distill-Llama-70B-gptq-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 70 --model-format gptq --quantization ${quantization}


Model Spec 9 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}


Model Spec 10 (mlx, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 70
- **Quantizations:** 3bit, 4bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-R1-Distill-Llama-70B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Llama-70B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/okwinds/DeepSeek-R1-Distill-Llama-70B-MLX-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-distill-llama --size-in-billions 70 --model-format mlx --quantization ${quantization}

