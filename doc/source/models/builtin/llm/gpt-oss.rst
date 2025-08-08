.. _models_llm_gpt-oss:

========================================
gpt-oss
========================================

- **Context Length:** 131072
- **Model Name:** gpt-oss
- **Languages:** en
- **Abilities:** chat
- **Description:** gpt-oss series, OpenAI’s open-weight models designed for powerful reasoning, agentic tasks, and versatile developer use cases.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 20 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 20
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** openai/gpt-oss-20b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openai/gpt-oss-20b>`__, `ModelScope <https://modelscope.cn/models/openai-mirror/gpt-oss-20b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gpt-oss --size-in-billions 20 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 120 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 120
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** openai/gpt-oss-120b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openai/gpt-oss-120b>`__, `ModelScope <https://modelscope.cn/models/openai-mirror/gpt-oss-120b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gpt-oss --size-in-billions 120 --model-format pytorch --quantization ${quantization}

