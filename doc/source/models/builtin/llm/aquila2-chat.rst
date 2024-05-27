.. _models_llm_aquila2-chat:

========================================
aquila2-chat
========================================

- **Context Length:** 2048
- **Model Name:** aquila2-chat
- **Languages:** zh
- **Abilities:** chat
- **Description:** Aquila2-chat series models are the chat models

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** BAAI/AquilaChat2-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/AquilaChat2-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/AquilaChat2-34B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/AquilaChat2-34B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (gptq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 34
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** TheBloke/AquilaChat2-34B-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/AquilaChat2-34B-GPTQ>`__, `ModelScope <https://modelscope.cn/models/BAAI/AquilaChat2-34B-Int4-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat --size-in-billions 34 --model-format gptq --quantization ${quantization}


Model Spec 4 (awq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 34
- **Quantizations:** Int4
- **Engines**: 
- **Model ID:** TheBloke/AquilaChat2-34B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/AquilaChat2-34B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat --size-in-billions 34 --model-format awq --quantization ${quantization}


Model Spec 5 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** BAAI/AquilaChat2-34B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/AquilaChat2-34B>`__, `ModelScope <https://modelscope.cn/models/BAAI/AquilaChat2-34B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** BAAI/AquilaChat2-70B-Expr
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/AquilaChat2-70B-Expr>`__, `ModelScope <https://modelscope.cn/models/BAAI/AquilaChat2-70B-Expr>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name aquila2-chat --size-in-billions 70 --model-format pytorch --quantization ${quantization}

