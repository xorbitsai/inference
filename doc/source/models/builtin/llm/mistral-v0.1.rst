.. _models_llm_mistral-v0.1:

========================================
mistral-v0.1
========================================

- **Context Length:** 8192
- **Model Name:** mistral-v0.1
- **Languages:** en
- **Abilities:** generate
- **Description:** Mistral-7B is a unmoderated Transformer based LLM claiming to outperform Llama2 on all benchmarks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** mistralai/Mistral-7B-v0.1
- **Model hub**: 

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mistral-v0.1 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Model ID:** TheBloke/Mistral-7B-v0.1-GGUF
- **Model hub**: 

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mistral-v0.1 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

