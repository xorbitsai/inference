.. _models_llm_hy-mt3-30b-a3b:

========================================
Hy-MT3-30B-A3B
========================================

- **Context Length:** 8192
- **Model Name:** Hy-MT3-30B-A3B
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Hy-MT3-30B-A3B is a mixture-of-experts multilingual translation model from Tencent Hunyuan. It supports 102 languages and follows translation instructions.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** tencent/Hy-MT3-30B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT3-30B-A3B>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT3-30B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT3-30B-A3B --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM, SGLang
- **Model ID:** tencent/Hy-MT3-30B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT3-30B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT3-30B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT3-30B-A3B --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 3 (ggufv2, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 30
- **Quantizations:** Q4_K_M, Q8_0, IQ2
- **Engines**: llama.cpp
- **Model ID:** tencent/Hy-MT3-30B-A3B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT3-30B-A3B-GGUF>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT3-30B-A3B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT3-30B-A3B --size-in-billions 30 --model-format ggufv2 --quantization ${quantization}
