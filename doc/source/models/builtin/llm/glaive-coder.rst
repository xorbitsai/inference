.. _models_llm_glaive-coder:

========================================
glaive-coder
========================================

- **Context Length:** 16384
- **Model Name:** glaive-coder
- **Languages:** en
- **Abilities:** chat
- **Description:** A code model trained on a dataset of ~140k programming related problems and solutions generated from Glaive’s synthetic data generation platform.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** glaiveai/glaive-coder-7b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/glaiveai/glaive-coder-7b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glaive-coder --size-in-billions 7 --model-format pytorch --quantization ${quantization}

