.. _models_llm_qwen3.5:

========================================
qwen3.5
========================================

- **Context Length:** 262144
- **Model Name:** qwen3.5
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning
- **Description:** Over recent months, we have intensified our focus on developing foundation models that deliver exceptional utility and performance. Qwen3.5 represents a significant leap forward, integrating breakthroughs in multimodal learning, architectural efficiency, reinforcement learning scale, and global accessibility to empower developers and enterprises with unprecedented capability and efficiency.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 397
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-397B-A17B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 397
- **Quantizations:** FP8
- **Engines**: 
- **Model ID:** Qwen/Qwen3.5-397B-A17B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format fp8 --quantization ${quantization}


Model Spec 3 (gptq, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 397
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 397 --model-format gptq --quantization ${quantization}


Model Spec 4 (pytorch, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 122
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-122B-A10B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-122B-A10B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format pytorch --quantization ${quantization}


Model Spec 5 (fp8, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 122
- **Quantizations:** FP8
- **Engines**: 
- **Model ID:** Qwen/Qwen3.5-122B-A10B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format fp8 --quantization ${quantization}


Model Spec 6 (gptq, 122 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 122
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 122 --model-format gptq --quantization ${quantization}


Model Spec 7 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-35B-A3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 8 (fp8, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 35
- **Quantizations:** FP8
- **Engines**: 
- **Model ID:** Qwen/Qwen3.5-35B-A3B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format fp8 --quantization ${quantization}


Model Spec 9 (gptq, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 35
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 35 --model-format gptq --quantization ${quantization}


Model Spec 10 (pytorch, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 27
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-27B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-27B>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-27B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format pytorch --quantization ${quantization}


Model Spec 11 (fp8, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 27
- **Quantizations:** FP8
- **Engines**: 
- **Model ID:** Qwen/Qwen3.5-27B-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-27B-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-27B-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format fp8 --quantization ${quantization}


Model Spec 12 (gptq, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 27
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen3.5-27B-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3.5-27b-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen3.5 --size-in-billions 27 --model-format gptq --quantization ${quantization}

