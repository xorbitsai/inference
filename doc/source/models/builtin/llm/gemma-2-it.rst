.. _models_llm_gemma-2-it:

========================================
gemma-2-it
========================================

- **Context Length:** 8192
- **Model Name:** gemma-2-it
- **Languages:** en
- **Abilities:** chat
- **Description:** Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none, 4-bit, 8-bit
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** google/gemma-2-2b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-2-2b-it>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/gemma-2-2b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none, 4-bit, 8-bit
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** google/gemma-2-9b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-2-9b-it>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/gemma-2-9b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 27
- **Quantizations:** none, 4-bit, 8-bit
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** google/gemma-2-27b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-2-27b-it>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/gemma-2-27b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 27 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** Q3_K_L, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0, f32
- **Engines**: llama.cpp
- **Model ID:** bartowski/gemma-2-2b-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/gemma-2-2b-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 9
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q4_K_L, Q4_K_M, Q4_K_S, Q5_K_L, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0, f32
- **Engines**: llama.cpp
- **Model ID:** bartowski/gemma-2-9b-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/gemma-2-9b-it-GGUF>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/gemma-2-9b-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 9 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 27
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q4_K_L, Q4_K_M, Q4_K_S, Q5_K_L, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0, f32
- **Engines**: llama.cpp
- **Model ID:** bartowski/gemma-2-27b-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/gemma-2-27b-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 27 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-2b-it-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-2b-it-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 8 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-2b-it-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-2b-it-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 9 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** None
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-2b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-2b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 10 (mlx, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 9
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-9b-it-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-9b-it-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 9 --model-format mlx --quantization ${quantization}


Model Spec 11 (mlx, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 9
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-9b-it-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-9b-it-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 9 --model-format mlx --quantization ${quantization}


Model Spec 12 (mlx, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 9
- **Quantizations:** None
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-9b-it-fp16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-9b-it-fp16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 9 --model-format mlx --quantization ${quantization}


Model Spec 13 (mlx, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 27
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-27b-it-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-27b-it-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 27 --model-format mlx --quantization ${quantization}


Model Spec 14 (mlx, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 27
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-27b-it-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-27b-it-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 27 --model-format mlx --quantization ${quantization}


Model Spec 15 (mlx, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 27
- **Quantizations:** None
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-2-27b-it-fp16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-2-27b-it-fp16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-2-it --size-in-billions 27 --model-format mlx --quantization ${quantization}

