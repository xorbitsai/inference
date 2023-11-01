.. _models_builtin_zephyr_7b_beta:

==============
Zephyr-7B-beta
==============

- **Context Length:** 8192
- **Model Name:** zephyr-7b-beta
- **Languages:** en
- **Abilities:** chat
- **Description:** Zephyr-7B-β is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1.

Specifications
^^^^^^^^^^^^^^

Model Spec (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** HuggingFaceH4/zephyr-7b-beta
- **Model Revision:** 3bac358730f8806e5c3dc7c7e19eb36e045bf720

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name zephyr-7b-beta --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
