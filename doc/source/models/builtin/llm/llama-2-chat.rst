.. _models_llm_llama-2-chat:

========================================
llama-2-chat
========================================

- **Context Length:** 4096
- **Model Name:** llama-2-chat
- **Languages:** en
- **Abilities:** chat
- **Description:** Llama-2-Chat is a fine-tuned version of the Llama-2 LLM, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-7B-Chat-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}


Model Spec 2 (ggmlv3, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 13
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-13B-chat-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 13 --model-format ggmlv3 --quantization ${quantization}


Model Spec 3 (ggmlv3, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 70
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Model ID:** TheBloke/Llama-2-70B-Chat-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 70 --model-format ggmlv3 --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Llama-2-7b-chat-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__, `ModelScope <https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Llama-2-13b-chat-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-13b-chat-hf>`__, `ModelScope <https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Llama-2-70b-chat-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-70b-chat-hf>`__, `ModelScope <https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 7 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Model ID:** TheBloke/Llama-2-7B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/Xorbits/Llama-2-7b-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 8 (ggufv2, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 13
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Model ID:** TheBloke/Llama-2-13B-chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/Xorbits/Llama-2-13b-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 13 --model-format ggufv2 --quantization ${quantization}


Model Spec 9 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Model ID:** TheBloke/Llama-2-70B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-2-chat --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}

