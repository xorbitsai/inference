.. _models_builtin_opt:

===
OPT
===

- **Model Name:** opt
- **Languages:** en
- **Abilities:** embed, generate

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 1 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** facebook/opt-125m

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name opt --size-in-billions 1 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
