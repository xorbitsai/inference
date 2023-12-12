.. _models_llm_qwen-chat:

========================================
qwen-chat
========================================

- **Context Length:** 2048
- **Model Name:** qwen-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Qwen-chat is a fine-tuned version of the Qwen LLM trained with alignment techniques, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q4_0
- **Model ID:** Xorbits/qwen-chat-7B-ggml

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}


Model Spec 2 (ggmlv3, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 14
- **Quantizations:** q4_0
- **Model ID:** Xorbits/qwen-chat-14B-ggml

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format ggmlv3 --quantization ${quantization}


Model Spec 3 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen-7B-Chat

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Model ID:** Qwen/Qwen-14B-Chat

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen-72B-Chat

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 6 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen-7B-Chat-{quantization}

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 7 (gptq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen-14B-Chat-{quantization}

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format gptq --quantization ${quantization}


Model Spec 8 (gptq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 72
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen-72B-Chat-{quantization}

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 72 --model-format gptq --quantization ${quantization}

