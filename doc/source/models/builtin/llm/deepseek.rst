.. _models_llm_deepseek:

========================================
deepseek
========================================

- **Context Length:** 4096
- **Model Name:** deepseek
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** DeepSeek LLM, trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vllm only available for quantization none)
- **Model ID:** deepseek-ai/deepseek-llm-7b-base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-llm-7b-base>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-llm-7b-base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 67 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 67
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vllm only available for quantization none)
- **Model ID:** deepseek-ai/deepseek-llm-67b-base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-llm-67b-base>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-llm-67b-base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek --size-in-billions 67 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/deepseek-llm-7B-chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-llm-7B-chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (ggufv2, 67 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 67
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/deepseek-llm-67b-chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/deepseek-llm-67b-chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek --size-in-billions 67 --model-format ggufv2 --quantization ${quantization}

