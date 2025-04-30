.. _models_llm_qwen3:

========================================
qwen3
========================================

- **Context Length:** 40960
- **Model Name:** qwen3
- **Languages:** en, zh
- **Abilities:** chat, reasoning, tools
- **Description:** Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_6
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-0.6B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-0.6B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-0.6B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 0_6
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-0.6B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-0.6B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-0.6B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format fp8 --quantization ${quantization}


Model Spec 3 (ggufv2, 0_6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 0_6
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-0.6B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-0.6B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 0_6 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (pytorch, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_7
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-1.7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-1.7B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-1.7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (fp8, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 1_7
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-1.7B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-1.7B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-1.7B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format fp8 --quantization ${quantization}


Model Spec 6 (ggufv2, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_7
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-1.7B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-1.7B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 1_7 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-4B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 8 (fp8, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 4
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-4B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-4B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-4B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format fp8 --quantization ${quantization}


Model Spec 9 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-4B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-4B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-4B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}


Model Spec 10 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-8B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 11 (fp8, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 8
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-8B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-8B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-8B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format fp8 --quantization ${quantization}


Model Spec 12 (ggufv2, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 8
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-8B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-8B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-8B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 8 --model-format ggufv2 --quantization ${quantization}


Model Spec 13 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-14B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-14B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-14B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 14 (fp8, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 14
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-14B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-14B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-14B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format fp8 --quantization ${quantization}


Model Spec 15 (ggufv2, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 14
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-14B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-14B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-14B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 14 --model-format ggufv2 --quantization ${quantization}


Model Spec 16 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-30B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-30B-A3B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-30B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 17 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-30B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-30B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 18 (ggufv2, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 30
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-30B-A3B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-30B-A3B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 30 --model-format ggufv2 --quantization ${quantization}


Model Spec 19 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-32B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 20 (fp8, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 32
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-32B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-32B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-32B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format fp8 --quantization ${quantization}


Model Spec 21 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-32B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-32B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-32B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 22 (pytorch, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 235
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3-235B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-235B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-235B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 235 --model-format pytorch --quantization ${quantization}


Model Spec 23 (fp8, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 235
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3-235B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-235B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-235B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 235 --model-format fp8 --quantization ${quantization}


Model Spec 24 (ggufv2, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 235
- **Quantizations:** Q2_K, Q2_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q5_K_M, Q6_K, Q8_0, BF16, UD-Q2_K_XL, UD-Q3_K_XL, IQ4_NL, IQ4_XS
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3-235B-A22B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3-235B-A22B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3 --size-in-billions 235 --model-format ggufv2 --quantization ${quantization}

