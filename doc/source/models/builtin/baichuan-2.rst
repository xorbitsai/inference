.. _models_builtin_baichuan_2:

==========
Baichuan-2
==========

- **Context Length:** 4096
- **Model Name:** baichuan-2
- **Languages:** en, zh
- **Abilities:** embed, generate
- **Description:** Baichuan2 is an open-source Transformer based LLM that is trained on both Chinese and English data.

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan2-7B-Base

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan-2 --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   Not supported on macOS.

Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan2-13B-Base

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan-2 --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   Not supported on macOS.
