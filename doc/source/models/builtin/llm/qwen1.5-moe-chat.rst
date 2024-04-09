.. _models_llm_qwen1.5-moe-chat:

========================================
qwen1.5-moe-chat
========================================

- **Context Length:** 32768
- **Model Name:** qwen1.5-moe-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Qwen1.5-MoE is a transformer-based MoE decoder-only language model pretrained on a large amount of data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2_7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-MoE-A2.7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-MoE-A2.7B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-moe-chat --size-in-billions 2_7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 2_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 2_7
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-moe-chat --size-in-billions 2_7 --model-format gptq --quantization ${quantization}

