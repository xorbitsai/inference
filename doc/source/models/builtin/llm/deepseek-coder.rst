.. _models_llm_deepseek-coder:

========================================
deepseek-coder
========================================

- **Context Length:** 16384
- **Model Name:** deepseek-coder
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_3
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/deepseek-coder-1.3b-base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-coder-1.3b-base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 1_3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 6_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6_7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/deepseek-coder-6.7b-base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-coder-6.7b-base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 6_7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/deepseek-coder-7b-base-v1.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-coder-7b-base-v1.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 33 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 33
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/deepseek-coder-33b-base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-coder-33b-base>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-coder-33b-base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 33 --model-format pytorch --quantization ${quantization}


Model Spec 5 (ggufv2, 1_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_3
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/deepseek-coder-1.3b-base-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-1.3b-base-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 1_3 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 6_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 6_7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/deepseek-coder-6.7B-base-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 6_7 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** dagbs/deepseek-coder-7b-base-v1.5-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/dagbs/deepseek-coder-7b-base-v1.5-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 8 (ggufv2, 33 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 33
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/deepseek-coder-33B-base-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-33B-base-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 33 --model-format ggufv2 --quantization ${quantization}


Model Spec 9 (gptq, 1_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1_3
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/deepseek-coder-1.3b-base-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-1.3b-base-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 1_3 --model-format gptq --quantization ${quantization}


Model Spec 10 (gptq, 6_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 6_7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/deepseek-coder-6.7B-base-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 6_7 --model-format gptq --quantization ${quantization}


Model Spec 11 (gptq, 33 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 33
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/deepseek-coder-33B-base-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-33B-base-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 33 --model-format gptq --quantization ${quantization}


Model Spec 12 (awq, 1_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1_3
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/deepseek-coder-1.3b-base-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-1.3b-base-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 1_3 --model-format awq --quantization ${quantization}


Model Spec 13 (awq, 6_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 6_7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/deepseek-coder-6.7B-base-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 6_7 --model-format awq --quantization ${quantization}


Model Spec 14 (awq, 33 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 33
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/deepseek-coder-33B-base-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-coder-33B-base-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-coder --size-in-billions 33 --model-format awq --quantization ${quantization}

