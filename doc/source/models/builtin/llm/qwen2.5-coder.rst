.. _models_llm_qwen2.5-coder:

========================================
qwen2.5-coder
========================================

- **Context Length:** 32768
- **Model Name:** qwen2.5-coder
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen).

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-0.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-0.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder --size-in-billions 0_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-1.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-1.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-3B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-3B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-3B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-7B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-14B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-14B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-14B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers, SGLang (vLLM and SGLang only available for quantization none)
- **Model ID:** Qwen/Qwen2.5-Coder-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-Coder-32B>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-Coder-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-coder --size-in-billions 32 --model-format pytorch --quantization ${quantization}

