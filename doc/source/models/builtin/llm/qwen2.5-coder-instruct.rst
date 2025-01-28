.. _models_llm_qwen2.5-coder-instruct:

========================================
qwen2.5-coder-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2.5-coder-instruct
- **Languages:** en, zh
- **Abilities:** chat, tools
- **Description:** Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen).

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-0.5B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-0.5B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 0_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-14B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-14B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-32B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-32B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 7 (gptq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 0_5 --model-format gptq --quantization ${quantization}


Model Spec 8 (gptq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 1_5 --model-format gptq --quantization ${quantization}


Model Spec 9 (gptq, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 3
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 3 --model-format gptq --quantization ${quantization}


Model Spec 10 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 11 (gptq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 14
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 14 --model-format gptq --quantization ${quantization}


Model Spec 12 (gptq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 32
- **Quantizations:** Int4, Int8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 32 --model-format gptq --quantization ${quantization}


Model Spec 13 (awq, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 0_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 0_5 --model-format awq --quantization ${quantization}


Model Spec 14 (awq, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1_5
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 1_5 --model-format awq --quantization ${quantization}


Model Spec 15 (awq, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 3
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-3B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-3B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 3 --model-format awq --quantization ${quantization}


Model Spec 16 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 17 (awq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-14B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-14B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 14 --model-format awq --quantization ${quantization}


Model Spec 18 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-32B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 32 --model-format awq --quantization ${quantization}


Model Spec 19 (ggufv2, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_5
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 1_5 --model-format ggufv2 --quantization ${quantization}


Model Spec 20 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** q2_k, q3_k_m, q4_0, q4_k_m, q5_0, q5_k_m, q6_k, q8_0
- **Engines**: llama.cpp
- **Model ID:** Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B-Instruct-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder-instruct --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

