.. _models_builtin_wizardcoder_python_v1_0:

=======================
WizardCoder-Python-v1.0
=======================

- **Context Length:** 100000
- **Model Name:** wizardcoder-python-v1.0
- **Languages:** en
- **Abilities:** generate, chat

Specifications
^^^^^^^^^^^^^^

Model Spec (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardCoder-Python-7B-V1.0

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.


Model Spec (pytorch, 13 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardCoder-Python-13B-V1.0

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec (pytorch, 34 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** WizardLM/WizardCoder-Python-34B-V1.0

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name wizardcoder-python-v1.0 --size-in-billions 34 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
