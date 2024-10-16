.. _models_llm_qwen2.5-instruct:

========================================
qwen2.5-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2.5-instruct
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-0.5B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-1.5B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-1.5B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-14B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-14B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-32B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-32B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 7 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-72B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-72B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 8 (gptq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-0.5B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format gptq --quantization ${quantization}


Model Spec 9 (gptq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-1.5B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-1.5B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 1_5 --model-format gptq --quantization ${quantization}


Model Spec 10 (gptq, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 3
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-3B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-3B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 3 --model-format gptq --quantization ${quantization}


Model Spec 11 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-7B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 12 (gptq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-14B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-14B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 14 --model-format gptq --quantization ${quantization}


Model Spec 13 (gptq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 32
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-32B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-32B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 32 --model-format gptq --quantization ${quantization}


Model Spec 14 (gptq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 72
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-72B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-72B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 72 --model-format gptq --quantization ${quantization}


Model Spec 15 (awq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-0.5B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format awq --quantization ${quantization}


Model Spec 16 (awq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-1.5B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-1.5B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 1_5 --model-format awq --quantization ${quantization}


Model Spec 17 (awq, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 3
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-3B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-3B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 3 --model-format awq --quantization ${quantization}


Model Spec 18 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-7B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 19 (awq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-14B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-14B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 14 --model-format awq --quantization ${quantization}


Model Spec 20 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-32B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-32B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 32 --model-format awq --quantization ${quantization}


Model Spec 21 (awq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 72
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-72B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-72B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 72 --model-format awq --quantization ${quantization}


Model Spec 22 (ggufv2, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 0_5
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-0.5B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 23 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-1.5B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-1.5B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 24 (ggufv2, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 3
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-3B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-3B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 3 --model-format ggufv2 --quantization ${quantization}


Model Spec 25 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-7B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}


Model Spec 26 (ggufv2, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 14
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-14B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-14B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 14 --model-format ggufv2 --quantization ${quantization}


Model Spec 27 (ggufv2, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 32
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-32B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-32B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 32 --model-format ggufv2 --quantization ${quantization}


Model Spec 28 (ggufv2, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 72
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0, fp16
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-72B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-72B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 72 --model-format ggufv2 --quantization ${quantization}


Model Spec 29 (mlx, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-0.5B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format mlx --quantization ${quantization}


Model Spec 30 (mlx, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 0_5
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-0.5B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format mlx --quantization ${quantization}


Model Spec 31 (mlx, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 0_5
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-0.5B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 0_5 --model-format mlx --quantization ${quantization}


Model Spec 32 (mlx, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-1.5B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 1_5 --model-format mlx --quantization ${quantization}


Model Spec 33 (mlx, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1_5
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-1.5B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 1_5 --model-format mlx --quantization ${quantization}


Model Spec 34 (mlx, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1_5
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-1.5B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 1_5 --model-format mlx --quantization ${quantization}


Model Spec 35 (mlx, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 3
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-3B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-3B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 3 --model-format mlx --quantization ${quantization}


Model Spec 36 (mlx, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 3
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-3B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-8bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-3B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 3 --model-format mlx --quantization ${quantization}


Model Spec 37 (mlx, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-3B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-3B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 3 --model-format mlx --quantization ${quantization}


Model Spec 38 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-7B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-7B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 7 --model-format mlx --quantization ${quantization}


Model Spec 39 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-7B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-8bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-7B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 7 --model-format mlx --quantization ${quantization}


Model Spec 40 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-7B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-7B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 7 --model-format mlx --quantization ${quantization}


Model Spec 41 (mlx, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 14
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-14B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-14B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 14 --model-format mlx --quantization ${quantization}


Model Spec 42 (mlx, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 14
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-14B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-8bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-14B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 14 --model-format mlx --quantization ${quantization}


Model Spec 43 (mlx, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-14B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-14B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 14 --model-format mlx --quantization ${quantization}


Model Spec 44 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-32B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-4bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-32B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 32 --model-format mlx --quantization ${quantization}


Model Spec 45 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-32B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-8bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-32B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 32 --model-format mlx --quantization ${quantization}


Model Spec 46 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-32B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-32B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-32B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 32 --model-format mlx --quantization ${quantization}


Model Spec 47 (mlx, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 72
- **Quantizations:** 4-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-72B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-72B-Instruct-4bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-72B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 72 --model-format mlx --quantization ${quantization}


Model Spec 48 (mlx, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 72
- **Quantizations:** 8-bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-72B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-72B-Instruct-8bit>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-72B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 72 --model-format mlx --quantization ${quantization}


Model Spec 49 (mlx, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 72
- **Quantizations:** none
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-72B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-72B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/okwinds/Qwen2.5-72B-Instruct-MLX-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-instruct --size-in-billions 72 --model-format mlx --quantization ${quantization}

