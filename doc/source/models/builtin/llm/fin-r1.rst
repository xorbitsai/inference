.. _models_llm_fin-r1:

========================================
fin-r1
========================================

- **Context Length:** 131072
- **Model Name:** fin-r1
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Fin-R1 is a large language model specifically designed for the field of financial reasoning

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** SUFE-AIFLM-Lab/Fin-R1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Fin-R1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name fin-r1 --size-in-billions 7 --model-format pytorch --quantization ${quantization}

