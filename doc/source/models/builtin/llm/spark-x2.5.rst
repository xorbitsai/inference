.. _models_llm_spark-x2.5:

========================================
Spark-X2.5
========================================

- **Context Length:** 1048576
- **Model Name:** Spark-X2.5
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning, hybrid
- **Description:** Spark-X2.5 is a compact general-purpose instruction model series with 1.7B and 4B checkpoints, native 1M-token context, reasoning, and tool-use support.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_7
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** XHToken/Spark-X2.5-1.7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-1.7B>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-1.7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 1_7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** XHToken/Spark-X2.5-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-4B>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_7
- **Quantizations:** Int8
- **Engines**: Transformers
- **Model ID:** XHToken/Spark-X2.5-1.7B-INT8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-1.7B-INT8>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-1.7B-INT8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 1_7 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** Int8
- **Engines**: Transformers
- **Model ID:** XHToken/Spark-X2.5-4B-INT8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-4B-INT8>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-4B-INT8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 5 (fp8, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 1_7
- **Quantizations:** FP8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** XHToken/Spark-X2.5-1.7B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-1.7B-FP8>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-1.7B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 1_7 --model-format fp8 --quantization ${quantization}


Model Spec 6 (fp8, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 4
- **Quantizations:** FP8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** XHToken/Spark-X2.5-4B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-4B-FP8>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-4B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 4 --model-format fp8 --quantization ${quantization}


Model Spec 7 (ggufv2, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1_7
- **Quantizations:** Q4_K_M, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** XHToken/Spark-X2.5-1.7B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-1.7B-GGUF>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-1.7B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 1_7 --model-format ggufv2 --quantization ${quantization}


Model Spec 8 (ggufv2, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 4
- **Quantizations:** Q4_K_M, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** XHToken/Spark-X2.5-4B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-4B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5 --size-in-billions 4 --model-format ggufv2 --quantization ${quantization}
