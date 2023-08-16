.. _models_builtin_wizardmath_v1_0:

===============
WizardMath v1.0
===============

- **Model Name:** wizardmath-v1.0
- **Languages:** en
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardMath-7B-V1.0

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardmath-v1.0 --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec (pytorch, 13 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardMath-13B-V1.0

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardmath-v1.0 --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec (pytorch, 70 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardMath-70B-V1.0

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardmath-v1.0 --size-in-billions 70 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
