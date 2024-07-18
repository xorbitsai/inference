.. _models_llm_codegeex4:

========================================
codegeex4
========================================

- **Context Length:** 131072
- **Model Name:** codegeex4
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** the open-source version of the latest CodeGeeX4 model series

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** THUDM/codegeex4-all-9b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/codegeex4-all-9b>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/codegeex4-all-9b>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codegeex4 --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 9
- **Quantizations:** IQ2_M, IQ3_M, Q4_K_M, Q5_K_M, Q6_K_L, Q8_0
- **Engines**: llama.cpp
- **Model ID:** THUDM/codegeex4-all-9b-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/codegeex4-all-9b-GGUF>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/codegeex4-all-9b-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codegeex4 --size-in-billions 9 --model-format ggufv2 --quantization ${quantization}

