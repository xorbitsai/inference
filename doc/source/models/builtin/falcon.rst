.. _models_builtin_falcon:

======
Falcon
======

- **Model Name:** falcon
- **Languages:** en
- **Abilities:** embed, generate

Specifications
^^^^^^^^^^^^^^

Model Spec 2 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** tiiuae/falcon-7b

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name falcon --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec 1 (pytorch, 40 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 40
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** tiiuae/falcon-40b

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name falcon --size-in-billions 40 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
