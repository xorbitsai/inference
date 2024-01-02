.. _models_llm_openhermes-2.5:

========================================
openhermes-2.5
========================================

- **Context Length:** 8192
- **Model Name:** openhermes-2.5
- **Languages:** en
- **Abilities:** chat
- **Description:** Openhermes 2.5 is a fine-tuned version of Mistral-7B-v0.1 on primarily GPT-4 generated data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** teknium/OpenHermes-2.5-Mistral-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name openhermes-2.5 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Model ID:** TheBloke/OpenHermes-2.5-Mistral-7B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name openhermes-2.5 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

