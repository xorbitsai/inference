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


Model Spec 1 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Engines**: 
- **Model ID:** TheBloke/Llama-2-7B-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 7 --model-format ggmlv3 --quantization ${quantization}


Model Spec 2 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: 
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
- **Engines**: 
- **Model ID:** TheBloke/Llama-2-7B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-7B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 4 (ggmlv3, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 13
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Engines**: 
- **Model ID:** TheBloke/Llama-2-13B-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-13B-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 13 --model-format ggmlv3 --quantization ${quantization}


Model Spec 5 (ggmlv3, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 70
- **Quantizations:** q2_K, q3_K_L, q3_K_M, q3_K_S, q4_0, q4_1, q4_K_M, q4_K_S, q5_0, q5_1, q5_K_M, q5_K_S, q6_K, q8_0
- **Engines**: 
- **Model ID:** TheBloke/Llama-2-70B-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-GGML>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 70 --model-format ggmlv3 --quantization ${quantization}


Model Spec 6 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: 
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
- **Engines**: 
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
- **Engines**: 
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
- **Engines**: 
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
- **Engines**: 
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
- **Engines**: 
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
- **Engines**: 
- **Model ID:** TheBloke/Llama-2-70B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Llama-2-70B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-2 --size-in-billions 70 --model-format awq --quantization ${quantization}

