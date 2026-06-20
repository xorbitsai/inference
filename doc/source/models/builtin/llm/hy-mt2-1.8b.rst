.. _models_llm_hy-mt2-1.8b:

========================================
Hy-MT2-1.8B
========================================

- **Context Length:** 262144
- **Model Name:** Hy-MT2-1.8B
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Hy-MT2-1.8B is a dense multilingual translation model from Tencent Hunyuan, supporting 33 languages with instruction-following capabilities for complex translation scenarios.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: Transformers, SGLang
- **Model ID:** tencent/Hy-MT2-1.8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT2-1.8B>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT2-1.8B --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** fp8
- **Engines**: Transformers
- **Model ID:** tencent/Hy-MT2-1.8B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT2-1.8B-FP8>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT2-1.8B --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** Q4_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** tencent/Hy-MT2-1.8B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-1.8B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT2-1.8B --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}

