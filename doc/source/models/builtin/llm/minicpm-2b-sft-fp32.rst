.. _models_llm_minicpm-2b-sft-fp32:

========================================
minicpm-2b-sft-fp32
========================================

- **Context Length:** 4096
- **Model Name:** minicpm-2b-sft-fp32
- **Languages:** zh
- **Abilities:** chat
- **Description:** MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Model ID:** openbmb/MiniCPM-2B-sft-fp32
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-2B-sft-fp32>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM-2B-sft-fp32>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name minicpm-2b-sft-fp32 --size-in-billions 2 --model-format pytorch --quantization ${quantization}

