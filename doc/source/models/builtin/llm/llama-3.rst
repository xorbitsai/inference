.. _models_llm_llama-3:

========================================
llama-3
========================================

- **Context Length:** 8192
- **Model Name:** llama-3
- **Languages:** en
- **Abilities:** generate
- **Description:** Llama 3 is an auto-regressive language model that uses an optimized transformer architecture

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Meta-Llama-3-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3-8B>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 8
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_0, Q5_1, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** QuantFactory/Meta-Llama-3-8B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3 --size-in-billions 8 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** meta-llama/Meta-Llama-3-70B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3-70B>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3-70B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3 --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** Q4_K_M, Q5_K_M
- **Engines**: llama.cpp
- **Model ID:** NousResearch/Meta-Llama-3-70B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/NousResearch/Meta-Llama-3-70B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3 --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}

