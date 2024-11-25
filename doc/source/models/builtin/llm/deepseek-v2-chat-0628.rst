.. _models_llm_deepseek-v2-chat-0628:

========================================
deepseek-v2-chat-0628
========================================

- **Context Length:** 128000
- **Model Name:** deepseek-v2-chat-0628
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** DeepSeek-V2-Chat-0628 is an improved version of DeepSeek-V2-Chat. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 236 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 236
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** deepseek-ai/DeepSeek-V2-Chat-0628
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Chat-0628>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v2-chat-0628 --size-in-billions 236 --model-format pytorch --quantization ${quantization}

