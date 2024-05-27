.. _models_llm_wizardlm-v1.0:

========================================
wizardlm-v1.0
========================================

- **Context Length:** 2048
- **Model Name:** wizardlm-v1.0
- **Languages:** en
- **Abilities:** chat
- **Description:** WizardLM is an open-source LLM trained by fine-tuning LLaMA with Evol-Instruct.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/WizardLM-7B-V1.0-Uncensored-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/WizardLM-7B-V1.0-Uncensored-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardlm-v1.0 --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}


Model Spec 2 (ggmlv3, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 13
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/WizardLM-13B-V1.0-Uncensored-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/WizardLM-13B-V1.0-Uncensored-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardlm-v1.0 --size-in-billions 13 --model-format ggmlv3 --quantization ${quantization}

