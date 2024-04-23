.. _models_llm_llama-3-instruct:

========================================
llama-3-instruct
========================================

- **Context Length:** 8192
- **Model Name:** llama-3-instruct
- **Languages:** en
- **Abilities:** chat
- **Description:** The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks..

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 8
- **Quantizations:** IQ3_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Model ID:** lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-3-instruct --size-in-billions 8 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Meta-Llama-3-8B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-3-instruct --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 3 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** IQ1_M, IQ2_XS, Q4_K_M
- **Model ID:** lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-3-instruct --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** meta-llama/Meta-Llama-3-70B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3-70B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name llama-3-instruct --size-in-billions 70 --model-format pytorch --quantization ${quantization}

