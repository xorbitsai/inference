.. _models_llm_skywork-or1-preview:

========================================
skywork-or1-preview
========================================

- **Context Length:** 32768
- **Model Name:** skywork-or1-preview
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** The Skywork-OR1 (Open Reasoner 1) model series consists of powerful math and code reasoning models trained using large-scale rule-based reinforcement learning with carefully designed datasets and training recipes.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Skywork/Skywork-OR1-32B-Preview
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Skywork/Skywork-OR1-32B-Preview>`__, `ModelScope <https://modelscope.cn/models/Skywork/Skywork-OR1-32B-Preview>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1-preview --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 32
- **Quantizations:** Int4, int8
- **Engines**: vLLM, Transformers
- **Model ID:** JunHowie/Skywork-OR1-32B-Preview-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Skywork-OR1-32B-Preview-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Skywork-OR1-32B-Preview-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1-preview --size-in-billions 32 --model-format gptq --quantization ${quantization}


Model Spec 3 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Skywork/Skywork-OR1-7B-Preview
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Skywork/Skywork-OR1-7B-Preview>`__, `ModelScope <https://modelscope.cn/models/Skywork/Skywork-OR1-7B-Preview>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1-preview --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** IQ2_M, IQ2_S, IQ2_XS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q3_K_XL, Q4_0, Q4_1, Q4_K_L, Q4_K_M, Q4_K_S, Q5_K_L, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** bartowski/Skywork_Skywork-OR1-32B-Preview-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/Skywork_Skywork-OR1-32B-Preview-GGUF>`__, `ModelScope <https://modelscope.cn/models/bartowski/Skywork_Skywork-OR1-32B-Preview-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1-preview --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** IQ2_M, IQ2_S, IQ2_XS, IQ3_M, IQ3_XS, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q2_K_L, Q3_K_L, Q3_K_M, Q3_K_S, Q3_K_XL, Q4_0, Q4_1, Q4_K_L, Q4_K_M, Q4_K_S, Q5_K_L, Q5_K_M, Q5_K_S, Q6_K, Q6_K_L, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** bartowski/Skywork_Skywork-OR1-7B-Preview-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/bartowski/Skywork_Skywork-OR1-7B-Preview-GGUF>`__, `ModelScope <https://modelscope.cn/models/bartowski/Skywork_Skywork-OR1-7B-Preview-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1-preview --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

