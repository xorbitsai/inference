.. _models_llm_c4ai-command-r-v01:

========================================
c4ai-command-r-v01
========================================

- **Context Length:** 131072
- **Model Name:** c4ai-command-r-v01
- **Languages:** en, fr, de, es, it, pt, ja, ko, zh, ar
- **Abilities:** chat
- **Description:** C4AI Command-R(+) is a research release of a 35 and 104 billion parameter highly performant generative model.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** CohereForAI/c4ai-command-r-v01
- **Model Hubs**:  `Hugging Face <https://huggingface.co/CohereForAI/c4ai-command-r-v01>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/c4ai-command-r-v01>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name c4ai-command-r-v01 --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 35
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: vLLM, Transformers
- **Model ID:** andrewcanis/c4ai-command-r-v01-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF>`__, `ModelScope <https://modelscope.cn/models/mirror013/C4AI-Command-R-v01-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name c4ai-command-r-v01 --size-in-billions 35 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (pytorch, 104 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 104
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** CohereForAI/c4ai-command-r-plus
- **Model Hubs**:  `Hugging Face <https://huggingface.co/CohereForAI/c4ai-command-r-plus>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/c4ai-command-r-plus>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name c4ai-command-r-v01 --size-in-billions 104 --model-format pytorch --quantization ${quantization}


Model Spec 4 (gptq, 104 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 104
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** alpindale/c4ai-command-r-plus-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/alpindale/c4ai-command-r-plus-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name c4ai-command-r-v01 --size-in-billions 104 --model-format gptq --quantization ${quantization}

