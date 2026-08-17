.. _models_llm_qwen3.8-max:

========================================
qwen3.8-max
========================================

- **Context Length:** 262144
- **Model Name:** qwen3.8-max
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning
- **Description:** Qwen3.8-2.4T-A95B is a text-only Mixture-of-Experts model with 2.4 trillion total parameters and 95 billion activated parameters, designed for coding, research, professional work, and long-horizon agentic tasks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2400 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2400
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen3.8-2.4T-A95B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.8-2.4T-A95B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8-max --size-in-billions 2400 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 2400 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 2400
- **Quantizations:** FP8
- **Engines**: vLLM, SGLang
- **Model ID:** Qwen/Qwen3.8-2.4T-A95B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.8-2.4T-A95B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8-max --size-in-billions 2400 --model-format fp8 --quantization ${quantization}


Model Spec 3 (ggufv2, 2400 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2400
- **Quantizations:** BF16, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_XS, UD-IQ2_XXS, UD-IQ3_XXS, UD-IQ4_XS, UD-Q1_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** unsloth/Qwen3.8-2.4T-A95B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/Qwen3.8-2.4T-A95B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8-max --size-in-billions 2400 --model-format ggufv2 --quantization ${quantization}

