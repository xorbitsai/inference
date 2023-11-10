.. _models_builtin_baichuan:

========
Baichuan
========

- **Model Name:** baichuan
- **Languages:** en, zh
- **Abilities:** embed, generate

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (ggmlv3)
+++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/baichuan-llama-7B-GGML

Execute the following command to launch the model, remember to replace ``${quantization}`` with your chosen quantization method from the options listed above::

   xinference launch --model-name baichuan --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}


.. note::

   For utilizing the Apple Metal GPU for acceleration, select the q4_0 and q4_1 quantizations.

Model Spec 2 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan-7B

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   Not supported on macOS.

Model Spec 3 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** baichuan-inc/Baichuan-13B-Base

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name baichuan --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   Not supported on macOS.
