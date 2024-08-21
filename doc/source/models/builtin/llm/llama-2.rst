.. _models_llm_llama-2:

========================================
llama-2
========================================

- **Context Length:** 4096
- **Model Name:** llama-2
- **Languages:** en
- **Abilities:** generate
- **Description:** Llama-2 is the second generation of Llama, open-source and trained on a larger amount of data.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/Llama-2-7B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, SGLang
- **Model ID:** TheBloke/Llama-2-7B-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, SGLang
- **Model ID:** TheBloke/Llama-2-7B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 4 (ggufv2, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 13
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/Llama-2-13B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 13 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 70
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M
- **Engines**: llama.cpp
- **Model ID:** TheBloke/Llama-2-70B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 70 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** meta-llama/Llama-2-7b-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-7b-hf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 7 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** meta-llama/Llama-2-13b-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-13b-hf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 13 --model-format pytorch --quantization ${quantization}


Model Spec 8 (gptq, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 13
- **Quantizations:** Int4
- **Engines**: vLLM, SGLang
- **Model ID:** TheBloke/Llama-2-13B-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 13 --model-format gptq --quantization ${quantization}


Model Spec 9 (awq, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 13
- **Quantizations:** Int4
- **Engines**: vLLM, SGLang
- **Model ID:** TheBloke/Llama-2-13B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 13 --model-format awq --quantization ${quantization}


Model Spec 10 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** meta-llama/Llama-2-70b-hf
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Llama-2-70b-hf>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 70 --model-format pytorch --quantization ${quantization}


Model Spec 11 (gptq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, SGLang
- **Model ID:** TheBloke/Llama-2-70B-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 70 --model-format gptq --quantization ${quantization}


Model Spec 12 (awq, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 70
- **Quantizations:** Int4
- **Engines**: vLLM, SGLang
- **Model ID:** TheBloke/Llama-2-70B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 70 --model-format awq --quantization ${quantization}

