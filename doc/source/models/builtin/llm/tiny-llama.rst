.. _models_llm_tiny-llama:

========================================
tiny-llama
========================================

- **Context Length:** 2048
- **Model Name:** tiny-llama
- **Languages:** en
- **Abilities:** generate
- **Description:** The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name tiny-llama --size-in-billions 1 --model-format ggufv2 --quantization ${quantization}

