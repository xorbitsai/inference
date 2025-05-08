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


Model Spec 1 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** WizardLMTeam/WizardCoder-Python-13B-V1.0
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WizardLMTeam/WizardCoder-Python-13B-V1.0>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/WizardCoder-Python-13B-V1.0>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardcoder-python-v1.0 --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** WizardLMTeam/WizardCoder-Python-34B-V1.0
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WizardLMTeam/WizardCoder-Python-34B-V1.0>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/WizardCoder-Python-34B-V1.0>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardcoder-python-v1.0 --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/WizardCoder-Python-7B-V1.0-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/WizardCoder-Python-7B-V1.0-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardcoder-python-v1.0 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (ggufv2, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 13
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/WizardCoder-Python-13B-V1.0-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/WizardCoder-Python-13B-V1.0-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardcoder-python-v1.0 --size-in-billions 13 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/WizardCoder-Python-34B-V1.0-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/WizardCoder-Python-34B-V1.0-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardcoder-python-v1.0 --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}

