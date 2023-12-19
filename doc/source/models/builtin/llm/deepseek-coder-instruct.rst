.. _models_llm_deepseek-coder-instruct:

========================================
deepseek-coder-instruct
========================================

- **Context Length:** 4096
- **Model Name:** deepseek-coder-instruct
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** deepseek-coder-instruct is a model initialized from deepseek-coder-base and fine-tuned on 2B tokens of instruction data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_3
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** deepseek-ai/deepseek-coder-1.3b-instruct

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-coder-instruct --size-in-billions 1_3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 6_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6_7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** deepseek-ai/deepseek-coder-6.7b-instruct

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-coder-instruct --size-in-billions 6_7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 33 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 33
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** deepseek-ai/deepseek-coder-33b-instruct

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-coder-instruct --size-in-billions 33 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 1_3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_3
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/deepseek-coder-1.3b-instruct-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-coder-instruct --size-in-billions 1_3 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 6_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 6_7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/deepseek-coder-6.7B-instruct-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-coder-instruct --size-in-billions 6_7 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 33 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 33
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/deepseek-coder-33B-instruct-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name deepseek-coder-instruct --size-in-billions 33 --model-format ggufv2 --quantization ${quantization}

