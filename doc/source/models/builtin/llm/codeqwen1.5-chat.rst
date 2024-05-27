.. _models_llm_codeqwen1.5-chat:

========================================
codeqwen1.5-chat
========================================

- **Context Length:** 65536
- **Model Name:** codeqwen1.5-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** CodeQwen1.5 is the Code-Specific version of Qwen1.5. It is a transformer-based decoder-only language model pretrained on a large amount of data of codes.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/CodeQwen1.5-7B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/CodeQwen1.5-7B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codeqwen1.5-chat --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vllm only available for quantization none)
- **Model ID:** Qwen/CodeQwen1.5-7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat>`__, `ModelScope <https://modelscope.cn/models/qwen/CodeQwen1.5-7B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codeqwen1.5-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/CodeQwen1.5-7B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/CodeQwen1.5-7B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codeqwen1.5-chat --size-in-billions 7 --model-format awq --quantization ${quantization}

