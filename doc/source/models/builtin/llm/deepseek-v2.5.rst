.. _models_llm_deepseek-v2.5:

========================================
deepseek-v2.5
========================================

- **Context Length:** 128000
- **Model Name:** deepseek-v2.5
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. The new model integrates the general and coding abilities of the two previous versions.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 236 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 236
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** deepseek-ai/DeepSeek-V2.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V2.5>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V2.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v2.5 --size-in-billions 236 --model-format pytorch --quantization ${quantization}

