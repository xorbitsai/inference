.. _models_builtin_llama_2_chat:

============
Llama-2 Chat
============

- **Model Name:** llama-2-chat
- **Languages:** en
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-7B-Chat-GGML

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}

Model Spec 2 (ggmlv3, 13 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 13
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-13B-chat-GGML

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 13 --model-format ggmlv3 --quantization ${quantization}


Model Spec 3 (ggmlv3, 70 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 70
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-70B-Chat-GGML

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 70 --model-format ggmlv3 --quantization ${quantization}

Model Spec 4 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Llama-2-7b-chat-hf

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.


Model Spec 5 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Llama-2-13b-chat-hf

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.

Model Spec 6 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Llama-2-70b-chat-hf

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 70 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.