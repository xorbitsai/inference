.. _models_llm_yi-1.5-chat-16k:

========================================
Yi-1.5-chat-16k
========================================

- **Context Length:** 16384
- **Model Name:** Yi-1.5-chat-16k
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** 01-ai/Yi-1.5-9B-Chat-16K
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-9B-Chat-16K>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-9B-Chat-16K>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi-1.5-chat-16k --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** 01-ai/Yi-1.5-34B-Chat-16K
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-34B-Chat-16K>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-34B-Chat-16K>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi-1.5-chat-16k --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 9
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** QuantFactory/Yi-1.5-9B-Chat-16K-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantFactory/Yi-1.5-9B-Chat-16K-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi-1.5-chat-16k --size-in-billions 9 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** bartowski/Yi-1.5-34B-Chat-16K-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/Yi-1.5-34B-Chat-16K-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi-1.5-chat-16k --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}

