.. _models_llm_wizardcoder-python-v1.0:

========================================
wizardcoder-python-v1.0
========================================

- **Context Length:** 100000
- **Model Name:** wizardcoder-python-v1.0
- **Languages:** en
- **Abilities:** chat
- **Description:** 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardCoder-Python-7B-V1.0

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardCoder-Python-13B-V1.0

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardCoder-Python-34B-V1.0

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/WizardCoder-Python-7B-V1.0-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 13
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/WizardCoder-Python-13B-V1.0-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 13 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/WizardCoder-Python-34B-V1.0-GGUF

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}

