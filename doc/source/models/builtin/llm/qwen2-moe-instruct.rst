.. _models_llm_qwen2-moe-instruct:

========================================
qwen2-moe-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2-moe-instruct
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Qwen1.5-MoE is a transformer-based MoE decoder-only language model pretrained on a large amount of data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** Qwen/Qwen2-57B-A14B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-57B-A14B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-moe-instruct --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 2_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-moe-instruct --size-in-billions 14 --model-format gptq --quantization ${quantization}

