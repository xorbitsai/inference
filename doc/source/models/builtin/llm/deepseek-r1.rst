.. _models_llm_deepseek-r1:

========================================
deepseek-r1
========================================

- **Context Length:** 163840
- **Model Name:** deepseek-r1
- **Languages:** en, zh
- **Abilities:** chat, reasoning
- **Description:** DeepSeek-R1, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 671
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1 --size-in-billions 671 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 671
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** cognitivecomputations/DeepSeek-R1-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cognitivecomputations/DeepSeek-R1-AWQ>`__, `ModelScope <https://modelscope.cn/models/cognitivecomputations/DeepSeek-R1-awq>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1 --size-in-billions 671 --model-format awq --quantization ${quantization}


Model Spec 3 (ggufv2, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 671
- **Quantizations:** UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS, UD-Q2_K_XL, Q2_K, Q2_K_L, Q2_K_XS, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/DeepSeek-R1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/DeepSeek-R1-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/DeepSeek-R1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1 --size-in-billions 671 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (mlx, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 671
- **Quantizations:** 2bit, 3bit, 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-R1-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-R1-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/DeepSeek-R1-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1 --size-in-billions 671 --model-format mlx --quantization ${quantization}

