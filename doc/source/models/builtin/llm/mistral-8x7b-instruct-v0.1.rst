.. _models_llm_mistral-8x7b-instruct-v0.1:

========================================
mistral-8x7b-instruct-v0.1
========================================

- **Context Length:** 32768
- **Model Name:** mistral-8x7b-instruct-v0.1
- **Languages:** en
- **Abilities:** chat
- **Description:** Mistral-8x7B-Instruct is a fine-tuned version of the Mistral-8x7B LLM, specializing in chatting. According to the blog of the Mistral team: "On MT-Bench, it reaches a score of 8.30, making it the best open-source model (As of December 11, 2023), with a performance comparable to GPT3.5."

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 12 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** mistralai/Mixtral-8x7B-Instruct-v0.1

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mistral-8x7b-instruct-v0.1 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Model ID:** TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name mistral-8x7b-instruct-v0.1 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

