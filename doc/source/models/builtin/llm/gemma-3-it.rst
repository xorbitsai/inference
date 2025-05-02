.. _models_llm_gemma-3-it:

========================================
gemma-3-it
========================================

- **Context Length:** 131072
- **Model Name:** gemma-3-it
- **Languages:** en
- **Abilities:** chat, vision
- **Description:** Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** google/gemma-3-4b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-3-4b-it>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/gemma-3-4b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 12
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** google/gemma-3-12b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-3-12b-it>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/gemma-3-12b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 12 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 27
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** google/gemma-3-27b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-3-27b-it>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/gemma-3-27b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 27 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_L, Q4_K_M, Q4_K_S, Q5_K_L, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0, bf16
- **Engines**: llama.cpp
- **Model ID:** bartowski/google_gemma-3-4b-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF>`__, `ModelScope <https://modelscope.cn/models/bartowski/google_gemma-3-4b-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 12
- **Quantizations:** IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_L, Q4_K_M, Q4_K_S, Q5_K_L, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0, bf16
- **Engines**: llama.cpp
- **Model ID:** bartowski/google_gemma-3-12b-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF>`__, `ModelScope <https://modelscope.cn/models/bartowski/google_gemma-3-12b-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 12 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 27
- **Quantizations:** IQ2_M, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_L, Q4_K_M, Q4_K_S, Q5_K_L, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0, bf16
- **Engines**: llama.cpp
- **Model ID:** bartowski/google_gemma-3-27b-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/google_gemma-3-27b-it-GGUF>`__, `ModelScope <https://modelscope.cn/models/bartowski/google_gemma-3-27b-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 27 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 4bit, 6bit, 8bit, fp16
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-3-4b-it-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-3-4b-it-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/gemma-3-4b-it-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 8 (mlx, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 12
- **Quantizations:** 4bit, 6bit, 8bit, fp16
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-3-12b-it-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-3-12b-it-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/gemma-3-12b-it-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 12 --model-format mlx --quantization ${quantization}


Model Spec 9 (mlx, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 27
- **Quantizations:** 4bit, 6bit, 8bit, fp16
- **Engines**: MLX
- **Model ID:** mlx-community/gemma-3-27b-it-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-3-27b-it-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/gemma-3-27b-it-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-3-it --size-in-billions 27 --model-format mlx --quantization ${quantization}

