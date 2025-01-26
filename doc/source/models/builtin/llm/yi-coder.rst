.. _models_llm_yi-coder:

========================================
yi-coder
========================================

- **Context Length:** 131072
- **Model Name:** yi-coder
- **Languages:** en
- **Abilities:** generate
- **Description:** Yi-Coder is a series of open-source code language models that delivers state-of-the-art coding performance with fewer than 10 billion parameters.Excelling in long-context understanding with a maximum context length of 128K tokens.Supporting 52 major programming languages, including popular ones such as Java, Python, JavaScript, and C++.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** 01-ai/Yi-Coder-9B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-Coder-9B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-Coder-9B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name yi-coder --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** 01-ai/Yi-Coder-1.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-Coder-1.5B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-Coder-1.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name yi-coder --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}

