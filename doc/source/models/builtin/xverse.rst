_models_builtin_xverse:

======
XVERSE
======

- **Context Length:** 2048
- **Model Name:** xverse
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** XVERSE is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology.

Specifications
^^^^^^^^^^^^^^

Model Specs (pytorch, Billions)
+++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none

XVERSE Model Variants
---------------------

1. XVERSE-7B
    - **Model ID:** xverse/XVERSE-7B
    - **Model Revision:** 3778b254def675586e9218ccb15b78d6ef66a3a7

2. XVERSE-13B
    - **Model ID:** xverse/XVERSE-13B
    - **Model Revision:** 11ac840dda17af81046614229fdd0c658afff747

3. XVERSE-65B
    - **Model ID:** xverse/XVERSE-65B
    - **Model Revision:** 7f1b7394f74c630f50612a19ba90bd021c373989

To launch a specific XVERSE model, use the following command and replace `${quantization}` with your chosen quantization method:
chosen quantization method from the options listed above and the size::

   xinference launch --model-name xverse --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Details
^^^^^^^^^^^^^

- **Version:** 1
- **Context Length:** 2048
- **Model Name:** xverse
- **Model Languages:** en, zh
- **Model Abilities:** generate
- **Model Description:** XVERSE is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology.

