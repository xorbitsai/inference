.. _models_llm_qwen3.8-flash-next:

========================================
qwen3.8-Flash-Next
========================================

- **Context Length:** 262144
- **Model Name:** qwen3.8-Flash-Next
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Qwen3.8-Flash-Next is a native vision-language Mixture-of-Experts model with 125B main-model parameters, 6B activated per token, and an additional 51B n-gram embedding table. Its Qwen4 experimental architecture combines Gated DeltaNet, Qwen Sparse Attention, gated residuals, and flexible thinking control.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 125 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 125
- **Quantizations:** none
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3.8-Flash-Next
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8-Flash-Next --size-in-billions 125 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 125 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 125
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3.8-Flash-Next-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.8-Flash-Next --size-in-billions 125 --model-format fp8 --quantization ${quantization}

