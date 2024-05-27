.. _models_llm_baichuan-2-chat:

========================================
baichuan-2-chat
========================================

- **Context Length:** 4096
- **Model Name:** baichuan-2-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Baichuan2-chat is a fine-tuned version of the Baichuan LLM, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vllm only available for quantization none)
- **Model ID:** baichuan-inc/Baichuan2-7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat>`__, `ModelScope <https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name baichuan-2-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vllm only available for quantization none)
- **Model ID:** baichuan-inc/Baichuan2-13B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat>`__, `ModelScope <https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name baichuan-2-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}

