.. _models_llm_code-llama-python:

========================================
code-llama-python
========================================

- **Context Length:** 100000
- **Model Name:** code-llama-python
- **Languages:** en
- **Abilities:** generate
- **Description:** Code-Llama-Python is a fine-tuned version of the Code-Llama LLM, specializing in Python.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/CodeLlama-7B-Python-fp16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/CodeLlama-7B-Python-fp16>`__, `ModelScope <https://modelscope.cn/models/Xorbits/CodeLlama-7B-Python-fp16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name code-llama-python --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/CodeLlama-13B-Python-fp16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/CodeLlama-13B-Python-fp16>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/CodeLlama-13b-Python-hf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name code-llama-python --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** TheBloke/CodeLlama-34B-Python-fp16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/CodeLlama-34B-Python-fp16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name code-llama-python --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/CodeLlama-7B-Python-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/CodeLlama-7B-Python-GGUF>`__, `ModelScope <https://modelscope.cn/models/Xorbits/CodeLlama-7B-Python-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name code-llama-python --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 13
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/CodeLlama-13B-Python-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/CodeLlama-13B-Python-GGUF>`__, `ModelScope <https://modelscope.cn/models/Xorbits/CodeLlama-13B-Python-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name code-llama-python --size-in-billions 13 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/CodeLlama-34B-Python-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/CodeLlama-34B-Python-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name code-llama-python --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}

