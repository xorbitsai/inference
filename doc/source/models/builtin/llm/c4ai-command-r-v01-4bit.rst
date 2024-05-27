.. _models_llm_c4ai-command-r-v01-4bit:

========================================
c4ai-command-r-v01-4bit
========================================

- **Context Length:** 131072
- **Model Name:** c4ai-command-r-v01-4bit
- **Languages:** en, fr, de, es, it, pt, ja, ko, zh, ar
- **Abilities:** generate
- **Description:** This model is 4bit quantized version of C4AI Command-R using bitsandbytes.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** CohereForAI/c4ai-command-r-v01-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/CohereForAI/c4ai-command-r-v01-4bit>`__, `ModelScope <https://modelscope.cn/models/mirror013/c4ai-command-r-v01-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name c4ai-command-r-v01-4bit --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 104 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 104
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** CohereForAI/c4ai-command-r-plus-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name c4ai-command-r-v01-4bit --size-in-billions 104 --model-format pytorch --quantization ${quantization}

