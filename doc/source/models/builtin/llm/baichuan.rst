.. _models_llm_baichuan:

========================================
baichuan
========================================

- **Context Length:** 4096
- **Model Name:** baichuan
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Baichuan is an open-source Transformer based LLM that is trained on both Chinese and English data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/baichuan-llama-7B-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/baichuan-llama-7B-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name baichuan --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** baichuan-inc/Baichuan-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name baichuan --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** baichuan-inc/Baichuan-13B-Base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/baichuan-inc/Baichuan-13B-Base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name baichuan --size-in-billions 13 --model-format pytorch --quantization ${quantization}

