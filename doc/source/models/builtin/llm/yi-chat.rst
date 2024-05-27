.. _models_llm_yi-chat:

========================================
Yi-chat
========================================

- **Context Length:** 4096
- **Model Name:** Yi-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** The Yi series models are large language models trained from scratch by developers at 01.AI.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (gptq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 34
- **Quantizations:** 8bits
- **Engines**: llama.cpp
- **Model ID:** 01-ai/Yi-34B-Chat-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-34B-Chat-{quantization}>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-34B-Chat-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-chat --size-in-billions 34 --model-format gptq --quantization ${quantization}


Model Spec 2 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: llama.cpp
- **Model ID:** 01-ai/Yi-6B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-6B-Chat>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-6B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-chat --size-in-billions 6 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: llama.cpp
- **Model ID:** 01-ai/Yi-34B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-34B-Chat>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-34B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-chat --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/Yi-34B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-chat --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}

