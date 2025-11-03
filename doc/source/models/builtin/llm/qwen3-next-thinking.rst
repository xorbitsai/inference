.. _models_llm_qwen3-next-thinking:

========================================
Qwen3-Next-Thinking
========================================

- **Context Length:** 262144
- **Model Name:** Qwen3-Next-Thinking
- **Languages:** en, zh
- **Abilities:** chat, reasoning, tools
- **Description:** Qwen3-Next-80B-A3B is the first installment in the Qwen3-Next series

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 80 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 80
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-Next-80B-A3B-Thinking
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Thinking>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Next-Thinking --size-in-billions 80 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 80 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 80
- **Quantizations:** fp8
- **Engines**: vLLM
- **Model ID:** Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Next-Thinking --size-in-billions 80 --model-format fp8 --quantization ${quantization}


Model Spec 3 (awq, 80 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 80
- **Quantizations:** 4bit, 8bit
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-Next-80B-A3B-Thinking-AWQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-Next-80B-A3B-Thinking-AWQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-Next-80B-A3B-Thinking-AWQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Next-Thinking --size-in-billions 80 --model-format awq --quantization ${quantization}


Model Spec 4 (mlx, 80 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 80
- **Quantizations:** 4bit, 5bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-Next-80B-A3B-Thinking-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Thinking-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-Next-80B-A3B-Thinking-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Next-Thinking --size-in-billions 80 --model-format mlx --quantization ${quantization}

