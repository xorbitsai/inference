.. _models_llm_minicpm-2b-sft-bf16:

========================================
minicpm-2b-sft-bf16
========================================

- **Context Length:** 4096
- **Model Name:** minicpm-2b-sft-bf16
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
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM-2B-sft-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/miniCPM-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm-2b-sft-bf16 --size-in-billions 2 --model-format pytorch --quantization ${quantization}

