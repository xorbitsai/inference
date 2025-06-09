.. _models_llm_glm-edge-chat:

========================================
glm-edge-chat
========================================

- **Context Length:** 8192
- **Model Name:** glm-edge-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** The GLM-Edge series is our attempt to face the end-side real-life scenarios, which consists of two sizes of large-language dialogue models and multimodal comprehension models (GLM-Edge-1.5B-Chat, GLM-Edge-4B-Chat, GLM-Edge-V-2B, GLM-Edge-V-5B). Among them, the 1.5B / 2B model is mainly for platforms such as mobile phones and cars, and the 4B / 5B model is mainly for platforms such as PCs.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** THUDM/glm-edge-1.5b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-1.5b-chat>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-chat --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** THUDM/glm-edge-4b-chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-4b-chat>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-4b-chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-chat --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** Q4_0, Q4_1, Q4_K, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-1.5b-chat-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-1.5b-chat-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-chat --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** F16
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-1.5b-chat-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-1.5b-chat-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-1.5b-chat-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-chat --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** Q4_0, Q4_1, Q4_K, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-4b-chat-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-4b-chat-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-4b-chat-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-chat --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** F16
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-4b-chat-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-4b-chat-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-4b-chat-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-chat --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}

