.. _models_llm_kat-v1:

========================================
KAT-V1
========================================

- **Context Length:** 131072
- **Model Name:** KAT-V1
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Kwaipilot-AutoThink ranks first among all open-source models on LiveCodeBench Pro, a challenging benchmark explicitly designed to prevent data leakage, and even surpasses strong proprietary systems such as Seed and o3-mini.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 40 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 40
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Kwaipilot/KAT-V1-40B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Kwaipilot/KAT-V1-40B>`__, `ModelScope <https://modelscope.cn/models/Kwaipilot/KAT-V1-40B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name KAT-V1 --size-in-billions 40 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 40 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 40
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** QuantTrio/KAT-V1-40B-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/KAT-V1-40B-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/KAT-V1-40B-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name KAT-V1 --size-in-billions 40 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 40 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 40
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** QuantTrio/KAT-V1-40B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/KAT-V1-40B-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/KAT-V1-40B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name KAT-V1 --size-in-billions 40 --model-format awq --quantization ${quantization}

