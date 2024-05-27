.. _models_llm_gorilla-openfunctions-v2:

========================================
gorilla-openfunctions-v2
========================================

- **Context Length:** 4096
- **Model Name:** gorilla-openfunctions-v2
- **Languages:** en
- **Abilities:** chat
- **Description:** OpenFunctions is designed to extend Large Language Model (LLM) Chat Completion feature to formulate executable APIs call given natural language instructions and API context.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: llama.cpp
- **Model ID:** gorilla-llm/gorilla-openfunctions-v2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gorilla-openfunctions-v2 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K
- **Engines**: llama.cpp
- **Model ID:** gorilla-llm//gorilla-openfunctions-v2-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/gorilla-llm//gorilla-openfunctions-v2-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gorilla-openfunctions-v2 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

