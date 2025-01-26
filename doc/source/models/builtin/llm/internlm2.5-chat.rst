.. _models_llm_internlm2.5-chat:

========================================
internlm2.5-chat
========================================

- **Context Length:** 32768
- **Model Name:** internlm2.5-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** InternLM2.5 series of the InternLM model.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_8
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** internlm/internlm2_5-1_8b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2_5-1_8b-chat>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 1_8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** internlm/internlm2_5-7b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2_5-7b-chat>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 20 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 20
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** internlm/internlm2_5-20b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2_5-20b-chat>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-20b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 20 --model-format pytorch --quantization ${quantization}


Model Spec 4 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** ModelCloud/internlm-2.5-7b-chat-gptq-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/ModelCloud/internlm-2.5-7b-chat-gptq-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 5 (ggufv2, 1_8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_8
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** internlm/internlm2_5-1_8b-chat-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2_5-1_8b-chat-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 1_8 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** internlm/internlm2_5-7b-chat-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2_5-7b-chat-gguf>`__, `ModelScope <https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (ggufv2, 20 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 20
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** internlm/internlm2_5-20b-chat-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/internlm/internlm2_5-20b-chat-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 20 --model-format ggufv2 --quantization ${quantization}


Model Spec 8 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/internlm2_5-7b-chat-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/internlm2_5-7b-chat-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 7 --model-format mlx --quantization ${quantization}


Model Spec 9 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/internlm2_5-7b-chat-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/internlm2_5-7b-chat-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internlm2.5-chat --size-in-billions 7 --model-format mlx --quantization ${quantization}

