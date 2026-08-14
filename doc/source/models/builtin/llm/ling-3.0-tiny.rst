.. _models_llm_ling-3.0-tiny:

========================================
Ling-3.0-tiny
========================================

- **Context Length:** 131072
- **Model Name:** Ling-3.0-tiny
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning, hybrid
- **Description:** Ling-3.0-tiny is a lightweight hybrid-reasoning MoE model with 7.9B total parameters and 1.3B activated parameters per token.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7_9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7_9
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** inclusionAI/Ling-3.0-tiny
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-tiny>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-tiny --size-in-billions 7_9 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 7_9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 7_9
- **Quantizations:** FP8
- **Engines**: Transformers
- **Model ID:** inclusionAI/Ling-3.0-tiny-fp8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-tiny-fp8>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny-fp8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-tiny --size-in-billions 7_9 --model-format fp8 --quantization ${quantization}


Model Spec 3 (pytorch, 7_9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7_9
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** inclusionAI/Ling-3.0-tiny-int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/inclusionAI/Ling-3.0-tiny-int4>`__, `ModelScope <https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny-int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ling-3.0-tiny --size-in-billions 7_9 --model-format pytorch --quantization ${quantization}

