.. _models_llm_openbuddy:

========================================
OpenBuddy
========================================

- **Context Length:** 2048
- **Model Name:** OpenBuddy
- **Languages:** en
- **Abilities:** chat
- **Description:** OpenBuddy is a powerful open multilingual chatbot model aimed at global users.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggmlv3, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 13
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Model ID:** TheBloke/OpenBuddy-Llama2-13B-v11.1-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/OpenBuddy-Llama2-13B-v11.1-GGML>`__, `ModelScope <https://modelscope.cn/models/Xorbits/OpenBuddy-Llama2-13B-v11.1-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name OpenBuddy --size-in-billions 13 --model-format ggmlv3 --quantization ${quantization}

