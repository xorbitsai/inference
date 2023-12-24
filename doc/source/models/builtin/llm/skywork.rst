.. _models_llm_skywork:

========================================
Skywork
========================================

- **Context Length:** 4096
- **Model Name:** Skywork
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Skywork is a series of large models developed by the Kunlun Group · Skywork team.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 8-bit, none
- **Model ID:** skywork/Skywork-13B-base

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Skywork --size-in-billions 13 --model-format pytorch --quantization ${quantization}
