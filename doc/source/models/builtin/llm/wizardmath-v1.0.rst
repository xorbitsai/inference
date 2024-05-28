.. _models_llm_wizardmath-v1.0:

========================================
wizardmath-v1.0
========================================

- **Context Length:** 2048
- **Model Name:** wizardmath-v1.0
- **Languages:** en
- **Abilities:** chat
- **Description:** WizardMath is an open-source LLM trained by fine-tuning Llama2 with Evol-Instruct, specializing in math.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** WizardLM/WizardMath-7B-V1.0
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WizardLM/WizardMath-7B-V1.0>`__, `ModelScope <https://modelscope.cn/models/Xorbits/WizardMath-7B-V1.0>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardmath-v1.0 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** WizardLM/WizardMath-13B-V1.0
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WizardLM/WizardMath-13B-V1.0>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardmath-v1.0 --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** WizardLM/WizardMath-70B-V1.0
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WizardLM/WizardMath-70B-V1.0>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name wizardmath-v1.0 --size-in-billions 70 --model-format pytorch --quantization ${quantization}

