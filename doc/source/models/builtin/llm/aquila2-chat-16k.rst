.. _models_llm_aquila2-chat-16k:

========================================
aquila2-chat-16k
========================================

- **Context Length:** 16384
- **Model Name:** aquila2-chat-16k
- **Languages:** zh
- **Abilities:** chat
- **Description:** AquilaChat2-16k series models are the long-text chat models

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** BAAI/AquilaChat2-7B-16K
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/AquilaChat2-7B-16K>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat-16k --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: Transformers
- **Model ID:** TheBloke/AquilaChat2-34B-16K-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/AquilaChat2-34B-16K-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat-16k --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (gptq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 34
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** TheBloke/AquilaChat2-34B-16K-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/AquilaChat2-34B-16K-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat-16k --size-in-billions 34 --model-format gptq --quantization ${quantization}


Model Spec 4 (awq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 34
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** TheBloke/AquilaChat2-34B-16K-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/AquilaChat2-34B-16K-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat-16k --size-in-billions 34 --model-format awq --quantization ${quantization}


Model Spec 5 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** BAAI/AquilaChat2-34B-16K
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/AquilaChat2-34B-16K>`__, `ModelScope <https://modelscope.cn/models/BAAI/AquilaChat2-34B-16K>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat-16k --size-in-billions 34 --model-format pytorch --quantization ${quantization}

