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


Model Spec 1 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q4_K_M
- **Model ID:** Xorbits/Qwen-7B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Xorbits/Qwen-7B-Chat-GGUF>`_, `ModelScope <https://modelscope.cn/models/Xorbits/Qwen-7B-Chat-GGUF>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (ggufv2, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 14
- **Quantizations:** Q4_K_M
- **Model ID:** Xorbits/Qwen-14B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Xorbits/Qwen-14B-Chat-GGUF>`_, `ModelScope <https://modelscope.cn/models/Xorbits/Qwen-14B-Chat-GGUF>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (pytorch, 1_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_8
- **Quantizations:** none
- **Model ID:** Qwen/Qwen-1_8B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-1_8B-Chat>`_, `ModelScope <https://modelscope.cn/models/qwen/Qwen-1_8B-Chat>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 1_8 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Model ID:** Qwen/Qwen-7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-7B-Chat>`_, `ModelScope <https://modelscope.cn/models/qwen/Qwen-7B-Chat>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Model ID:** Qwen/Qwen-14B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-14B-Chat>`_, `ModelScope <https://modelscope.cn/models/qwen/Qwen-14B-Chat>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** none
- **Model ID:** Qwen/Qwen-72B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-72B-Chat>`_, `ModelScope <https://modelscope.cn/models/qwen/Qwen-72B-Chat>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 7 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen-7B-Chat-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-7B-Chat-{quantization}>`_, `ModelScope <https://modelscope.cn/models/qwen/Qwen-7B-Chat-{quantization}>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 8 (gptq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen-14B-Chat-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-14B-Chat-{quantization}>`_, `ModelScope <https://modelscope.cn/models/qwen/Qwen-14B-Chat-{quantization}>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format gptq --quantization ${quantization}


Model Spec 9 (gptq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 72
- **Quantizations:** Int4, Int8
- **Model ID:** Qwen/Qwen-72B-Chat-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen-72B-Chat-{quantization}>`_, `ModelScope <https://modelscope.cn/models/qwen/Qwen-72B-Chat-{quantization}>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 72 --model-format gptq --quantization ${quantization}

