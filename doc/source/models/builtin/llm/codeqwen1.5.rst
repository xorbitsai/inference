.. _models_llm_codeqwen1.5:

========================================
codeqwen1.5
========================================

- **Context Length:** 65536
- **Model Name:** codeqwen1.5
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** CodeQwen1.5 is the Code-Specific version of Qwen1.5. It is a transformer-based decoder-only language model pretrained on a large amount of data of codes.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/CodeQwen1.5-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/CodeQwen1.5-7B>`__, `ModelScope <https://modelscope.cn/models/qwen/CodeQwen1.5-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name codeqwen1.5 --size-in-billions 7 --model-format pytorch --quantization ${quantization}

