.. _models_llm_deepseek-v2-chat:

========================================
deepseek-v2-chat
========================================

- **Context Length:** 128000
- **Model Name:** deepseek-v2-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** deepseek-ai/DeepSeek-V2-Lite-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v2-chat --size-in-billions 16 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 236 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 236
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** deepseek-ai/DeepSeek-V2-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v2-chat --size-in-billions 236 --model-format pytorch --quantization ${quantization}

