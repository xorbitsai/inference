.. _models_builtin_code_llama_python:


=================
Code-Llama-Python
=================

- **Context Length:** 100000
- **Model Name:** code-llama-python
- **Languages:** en
- **Abilities:** generate

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** TheBloke/CodeLlama-7B-Python-fp16

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name code-llama-python --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** TheBloke/CodeLlama-13B-Python-fp16

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name code-llama-python --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec 3 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** TheBloke/CodeLlama-34B-Python-fp16

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name code-llama-python --size-in-billions 34 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
