.. _models_llm_qwen1.5-chat:

========================================
qwen1.5-chat
========================================

- **Context Length:** 32768
- **Model Name:** qwen1.5-chat
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-0.5B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 0_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 1_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_8
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-1.8B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 1_8 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-4B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-4B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-7B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-14B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-14B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-32B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-32B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-32B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 7 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-72B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-72B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-72B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 8 (pytorch, 110 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 110
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen1.5-110B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-110B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-110B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 110 --model-format pytorch --quantization ${quantization}


Model Spec 9 (gptq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen1.5-0.5B-Chat-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 0_5 --model-format gptq --quantization ${quantization}


Model Spec 10 (gptq, 1_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1_8
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen1.5-1.8B-Chat-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 1_8 --model-format gptq --quantization ${quantization}


Model Spec 11 (gptq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 4
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen1.5-4B-Chat-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-4B-Chat-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 4 --model-format gptq --quantization ${quantization}


Model Spec 12 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen1.5-7B-Chat-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 13 (gptq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen1.5-14B-Chat-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 14 --model-format gptq --quantization ${quantization}


Model Spec 14 (gptq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-32B-Chat-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-32B-Chat-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-32B-Chat-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 32 --model-format gptq --quantization ${quantization}


Model Spec 15 (gptq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 72
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen1.5-72B-Chat-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-72B-Chat-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 72 --model-format gptq --quantization ${quantization}


Model Spec 16 (gptq, 110 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 110
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-110B-Chat-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-110B-Chat-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-110B-Chat-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 110 --model-format gptq --quantization ${quantization}


Model Spec 17 (awq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-0.5B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 0_5 --model-format awq --quantization ${quantization}


Model Spec 18 (awq, 1_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1_8
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-1.8B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 1_8 --model-format awq --quantization ${quantization}


Model Spec 19 (awq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 4
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-4B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-4B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 4 --model-format awq --quantization ${quantization}


Model Spec 20 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-7B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-7B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 21 (awq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-14B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-14B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 14 --model-format awq --quantization ${quantization}


Model Spec 22 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-32B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-32B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-32B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 32 --model-format awq --quantization ${quantization}


Model Spec 23 (awq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 72
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-72B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-72B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-72B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 72 --model-format awq --quantization ${quantization}


Model Spec 24 (awq, 110 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 110
- **Quantizations:** Int4
- **Model ID:** Qwen/Qwen1.5-110B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-110B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-110B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 110 --model-format awq --quantization ${quantization}


Model Spec 25 (ggufv2, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 0_5
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Model ID:** Qwen/Qwen1.5-0.5B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-0.5B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 0_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 26 (ggufv2, 1_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_8
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Model ID:** Qwen/Qwen1.5-1.8B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-1.8B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 1_8 --model-format ggufv2 --quantization ${quantization}


Model Spec 27 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Model ID:** Qwen/Qwen1.5-4B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-4B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-4B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}


Model Spec 28 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Model ID:** Qwen/Qwen1.5-7B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 29 (ggufv2, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 14
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Model ID:** Qwen/Qwen1.5-14B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 14 --model-format ggufv2 --quantization ${quantization}


Model Spec 30 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Model ID:** Qwen/Qwen1.5-32B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-32B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-32B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 31 (ggufv2, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 72
- **Quantizations:** q2_k, q3_k_m, q4_k_m
- **Model ID:** Qwen/Qwen1.5-72B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen1.5-72B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen1.5-chat --size-in-billions 72 --model-format ggufv2 --quantization ${quantization}

