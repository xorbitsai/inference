.. _models_llm_ernie4.5:

========================================
Ernie4.5
========================================

- **Context Length:** 131072
- **Model Name:** Ernie4.5
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** ERNIE 4.5, a new family of large-scale multimodal models comprising 10 distinct variants.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_3
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** baidu/ERNIE-4.5-0.3B-PT
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT>`__, `ModelScope <https://modelscope.cn/models/PaddlePaddle/ERNIE-4.5-0.3B-PT>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 0_3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 0_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 0_3
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, F16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/ERNIE-4.5-0.3B-PT-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/ERNIE-4.5-0.3B-PT-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/ERNIE-4.5-0.3B-PT-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 0_3 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (mlx, 0_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 0_3
- **Quantizations:** 4bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/ERNIE-4.5-0.3B-PT-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/ERNIE-4.5-0.3B-PT-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/ERNIE-4.5-0.3B-PT-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 0_3 --model-format mlx --quantization ${quantization}


Model Spec 4 (pytorch, 21 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 21
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** baidu/ERNIE-4.5-21B-A3B-Base-PT
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Base-PT>`__, `ModelScope <https://modelscope.cn/models/PaddlePaddle/ERNIE-4.5-21B-A3B-Base-PT>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 21 --model-format pytorch --quantization ${quantization}


Model Spec 5 (ggufv2, 21 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 21
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, BF16
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/ERNIE-4.5-21B-A3B-PT-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/ERNIE-4.5-21B-A3B-PT-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/ERNIE-4.5-21B-A3B-PT-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 21 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (mlx, 21 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 21
- **Quantizations:** 4bit, 5bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/ERNIE-4.5-21B-A3B-PT-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/ERNIE-4.5-21B-A3B-PT-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/ERNIE-4.5-21B-A3B-PT-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 21 --model-format mlx --quantization ${quantization}


Model Spec 7 (pytorch, 300 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 300
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** baidu/ERNIE-4.5-300B-A47B-PT
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-PT>`__, `ModelScope <https://modelscope.cn/models/PaddlePaddle/ERNIE-4.5-300B-A47B-PT>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 300 --model-format pytorch --quantization ${quantization}


Model Spec 8 (ggufv2, 300 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 300
- **Quantizations:** Q2_K, Q4_K_M, Q6_K, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/ERNIE-4.5-300B-A47B-PT-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/ERNIE-4.5-300B-A47B-PT-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/ERNIE-4.5-300B-A47B-PT-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 300 --model-format ggufv2 --quantization ${quantization}


Model Spec 9 (mlx, 300 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 300
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/ERNIE-4.5-300B-47B-PT-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/ERNIE-4.5-300B-47B-PT-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/ERNIE-4.5-300B-47B-PT-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ernie4.5 --size-in-billions 300 --model-format mlx --quantization ${quantization}

