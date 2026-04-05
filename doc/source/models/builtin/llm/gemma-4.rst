.. _models_llm_gemma-4:

========================================
gemma-4
========================================

- **Context Length:** 262144
- **Model Name:** gemma-4
- **Languages:** en, zh
- **Abilities:** generate, chat, reasoning, audio, vision, hybrid
- **Description:** Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on small models) and generating text output.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** google/gemma-4-E2B-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-4-E2B-it>`__, `ModelScope <https://modelscope.cn/models/google/gemma-4-E2B-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** google/gemma-4-E4B-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-4-E4B-it>`__, `ModelScope <https://modelscope.cn/models/google/gemma-4-E4B-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 31 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 31
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** google/gemma-4-31B-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-4-31B-it>`__, `ModelScope <https://modelscope.cn/models/google/gemma-4-31B-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 31 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 26
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** google/gemma-4-26B-A4B-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-4-26B-A4B-it>`__, `ModelScope <https://modelscope.cn/models/google/gemma-4-26B-A4B-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 26 --model-format pytorch --quantization ${quantization}


Model Spec 5 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: 
- **Model ID:** unsloth/gemma-4-E2B-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/gemma-4-E2B-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: 
- **Model ID:** unsloth/gemma-4-E4B-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/gemma-4-E4B-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (ggufv2, 31 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 31
- **Quantizations:** IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: 
- **Model ID:** unsloth/gemma-4-31B-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/gemma-4-31B-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 31 --model-format ggufv2 --quantization ${quantization}


Model Spec 8 (ggufv2, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 26
- **Quantizations:** MXFP4_MOE, Q8_0, UD-IQ2_M, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_M, UD-Q3_K_S, UD-Q3_K_XL, UD-Q4_K_M, UD-Q4_K_S, UD-Q4_K_XL, UD-Q5_K_M, UD-Q5_K_S, UD-Q5_K_XL, UD-Q6_K, UD-Q6_K_XL, UD-Q8_K_XL
- **Engines**: 
- **Model ID:** unsloth/gemma-4-26B-A4B-it-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/gemma-4-26B-A4B-it-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 26 --model-format ggufv2 --quantization ${quantization}


Model Spec 9 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** bf16, 8bit, 6bit, 5bit, 4bit, mxfp8, mxfp4, nvfp4
- **Engines**: 
- **Model ID:** mlx-community/gemma-4-e2b-it-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-4-e2b-it-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/gemma-4-e2b-it-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 10 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** bf16, 8bit, 6bit, 5bit, 4bit, mxfp8, mxfp4, nvfp4
- **Engines**: 
- **Model ID:** mlx-community/gemma-4-e4b-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-4-e4b-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/gemma-4-e4b-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 11 (mlx, 31 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 31
- **Quantizations:** bf16, 8bit, 6bit, 5bit, 4bit, mxfp8, mxfp4, nvfp4
- **Engines**: 
- **Model ID:** mlx-community/gemma-4-31b-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-4-31b-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/gemma-4-31b-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 31 --model-format mlx --quantization ${quantization}


Model Spec 12 (mlx, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 26
- **Quantizations:** bf16, 8bit, 6bit, 5bit, 4bit, mxfp8, mxfp4, nvfp4
- **Engines**: 
- **Model ID:** mlx-community/gemma-4-26b-a4b-it-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/gemma-4-26b-a4b-it-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-4 --size-in-billions 26 --model-format mlx --quantization ${quantization}

