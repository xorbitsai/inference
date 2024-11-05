.. _models_llm_qwen2.5:

========================================
qwen2.5
========================================

- **Context Length:** 32768
- **Model Name:** qwen2.5
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-0.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-0.5B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-0.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5 --size-in-billions 0_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-1.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-1.5B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-1.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5 --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-3B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5 --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-7B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-14B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-14B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-14B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5 --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-32B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5 --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 7 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-72B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-72B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-72B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5 --size-in-billions 72 --model-format pytorch --quantization ${quantization}

