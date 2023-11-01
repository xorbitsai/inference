.. _models_builtin_orca_mini:

=========
Orca Mini
=========

- **Model Name:** orca
- **Languages:** en
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (ggmlv3, 3 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 3
- **Quantizations:** q4_0, q4_1, q5_0, q5_1, q8_0
- **Model ID:** TheBloke/orca_mini_3B-GGML

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name orca --size-in-billions 3 --model-format ggmlv3 --quantization ${quantization}

Model Spec 2 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q4_0, q4_1, q5_0, q5_1, q8_0
- **Model ID:** TheBloke/orca_mini_7B-GGML

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name orca --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}

Model Spec 3 (ggmlv3, 13 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 13
- **Quantizations:** q4_0, q4_1, q5_0, q5_1, q8_0
- **Model ID:** TheBloke/orca_mini_13B-GGML

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name orca --size-in-billions 13 --model-format ggmlv3 --quantization ${quantization}
