.. _models_llm_deepseek-v3.1:

========================================
Deepseek-V3.1
========================================

- **Context Length:** 131072
- **Model Name:** Deepseek-V3.1
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 671
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/DeepSeek-V3.1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V3.1>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Deepseek-V3.1 --size-in-billions 671 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 671
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/DeepSeek-V3.1-GPTQ-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/DeepSeek-V3.1-GPTQ-4bit>`__, `ModelScope <https://modelscope.cn/models/cpatonn/DeepSeek-V3.1-GPTQ-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Deepseek-V3.1 --size-in-billions 671 --model-format gptq --quantization ${quantization}


Model Spec 3 (awq, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 671
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/DeepSeek-V3.1-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/DeepSeek-V3.1-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/DeepSeek-V3.1-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Deepseek-V3.1 --size-in-billions 671 --model-format awq --quantization ${quantization}


Model Spec 4 (mlx, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 671
- **Quantizations:** 8bit, 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/DeepSeek-V3.1-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-V3.1-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/DeepSeek-V3.1-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Deepseek-V3.1 --size-in-billions 671 --model-format mlx --quantization ${quantization}

