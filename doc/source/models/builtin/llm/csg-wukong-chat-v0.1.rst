.. _models_llm_csg-wukong-chat-v0.1:

========================================
csg-wukong-chat-v0.1
========================================

- **Context Length:** 32768
- **Model Name:** csg-wukong-chat-v0.1
- **Languages:** en
- **Abilities:** chat
- **Description:** csg-wukong-1B is a 1 billion-parameter small language model(SLM) pretrained on 1T tokens.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** opencsg/csg-wukong-1B-chat-v0.1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/opencsg/csg-wukong-1B-chat-v0.1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name csg-wukong-chat-v0.1 --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1
- **Quantizations:** Q2_K, Q3_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** RichardErkhov/opencsg_-_csg-wukong-1B-chat-v0.1-gguf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/RichardErkhov/opencsg_-_csg-wukong-1B-chat-v0.1-gguf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name csg-wukong-chat-v0.1 --size-in-billions 1 --model-format ggufv2 --quantization ${quantization}

