.. _models_llm_internlm-7b:

========================================
internlm-7b
========================================

- **Context Length:** 8192
- **Model Name:** internlm-7b
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** InternLM is a Transformer-based LLM that is trained on both Chinese and English data, focusing on practical scenarios.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** internlm/internlm-7b
- **Model hub**: 

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name internlm-7b --size-in-billions 7 --model-format pytorch --quantization ${quantization}

