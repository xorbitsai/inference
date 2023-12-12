.. _models_llm_yi-chat:

========================================
Yi-chat
========================================

- **Context Length:** 204800
- **Model Name:** Yi-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** The Yi series models are large language models trained from scratch by developers at 01.AI.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** 01-ai/Yi-34B-Chat

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi-chat --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/Yi-34B-Chat-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi-chat --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}

