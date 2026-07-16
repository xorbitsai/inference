.. _models_llm_hy-mt2-30b-a3b:

========================================
Hy-MT2-30B-A3B
========================================

- **Context Length:** 262144
- **Model Name:** Hy-MT2-30B-A3B
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Hy-MT2-30B-A3B is a Mixture-of-Experts multilingual translation model from Tencent Hunyuan, supporting 33 languages with SOTA performance for complex professional domains.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: Transformers, SGLang
- **Model ID:** tencent/Hy-MT2-30B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT2-30B-A3B>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-30B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT2-30B-A3B --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: Transformers
- **Model ID:** tencent/Hy-MT2-30B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tencent/Hy-MT2-30B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/Tencent-Hunyuan/Hy-MT2-30B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Hy-MT2-30B-A3B --size-in-billions 30 --model-format pytorch --quantization ${quantization}

