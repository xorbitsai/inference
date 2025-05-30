.. _models_llm_deepseek-r1-0528:

========================================
deepseek-r1-0528
========================================

- **Context Length:** 163840
- **Model Name:** deepseek-r1-0528
- **Languages:** en, zh
- **Abilities:** chat, reasoning
- **Description:** DeepSeek-R1, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 671
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-0528
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-0528 --size-in-billions 671 --model-format pytorch --quantization ${quantization}

