.. _models_llm_glm-edge-v:

========================================
glm-edge-v
========================================

- **Context Length:** 8192
- **Model Name:** glm-edge-v
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** The GLM-Edge series is our attempt to face the end-side real-life scenarios, which consists of two sizes of large-language dialogue models and multimodal comprehension models (GLM-Edge-1.5B-Chat, GLM-Edge-4B-Chat, GLM-Edge-V-2B, GLM-Edge-V-5B). Among them, the 1.5B / 2B model is mainly for platforms such as mobile phones and cars, and the 4B / 5B model is mainly for platforms such as PCs.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** THUDM/glm-edge-v-2b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-2b>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-2b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** THUDM/glm-edge-v-5b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-5b>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-5b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 5 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** Q4_0, Q4_1, Q4_K, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-v-2b-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-2b-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-2b-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** F16
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-v-2b-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-2b-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-2b-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** f16
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-v-2b-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-2b-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-2b-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 5
- **Quantizations:** Q4_0, Q4_1, Q4_K, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-v-5b-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-5b-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-5b-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 5 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (ggufv2, 5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 5
- **Quantizations:** F16
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-v-5b-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-5b-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-5b-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 5 --model-format ggufv2 --quantization ${quantization}


Model Spec 8 (ggufv2, 5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 5
- **Quantizations:** f16
- **Engines**: llama.cpp
- **Model ID:** THUDM/glm-edge-v-5b-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/glm-edge-v-5b-gguf>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/glm-edge-v-5b-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-edge-v --size-in-billions 5 --model-format ggufv2 --quantization ${quantization}

