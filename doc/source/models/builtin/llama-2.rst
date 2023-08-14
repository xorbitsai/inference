.. _models_builtin_llama_2:

=======
Llama-2
=======

- **Model Name:** llama-2
- **Languages:** en
- **Abilities:** embed, generate

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-7B-GGML

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2 --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}

Model Spec 2 (ggmlv3, 13 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 13
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-13B-GGML

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2 --size-in-billions 13 --model-format ggmlv3 --quantization ${quantization}
