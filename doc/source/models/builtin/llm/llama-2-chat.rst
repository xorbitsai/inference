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


Model Spec 1 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** TheBloke/Llama-2-7B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/Xorbits/Llama-2-7b-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (ggufv2, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 13
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** TheBloke/Llama-2-13B-chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF>`__, `ModelScope <https://modelscope.cn/models/Xorbits/Llama-2-13b-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 13 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M
- **Engines**: vLLM, llama.cpp
- **Model ID:** TheBloke/Llama-2-70B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Llama-2-7b-chat-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__, `ModelScope <https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Llama-2-7B-Chat-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 6 (gptq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Llama-2-70B-Chat-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 70 --model-format gptq --quantization ${quantization}


Model Spec 7 (awq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Llama-2-70B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 70 --model-format awq --quantization ${quantization}


Model Spec 8 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Llama-2-7B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 9 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Llama-2-13b-chat-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-13b-chat-hf>`__, `ModelScope <https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 10 (gptq, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 13
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Llama-2-13B-chat-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 13 --model-format gptq --quantization ${quantization}


Model Spec 11 (awq, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 13
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** TheBloke/Llama-2-13B-chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 13 --model-format awq --quantization ${quantization}


Model Spec 12 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Llama-2-70b-chat-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-70b-chat-hf>`__, `ModelScope <https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2-chat --size-in-billions 70 --model-format pytorch --quantization ${quantization}

