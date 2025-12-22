.. _models_llm_qwen3-vl-thinking:

========================================
Qwen3-VL-Thinking
========================================

- **Context Length:** 262144
- **Model Name:** Qwen3-VL-Thinking
- **Languages:** en, zh
- **Abilities:** chat, vision, reasoning, tools
- **Description:** Meet Qwen3-VL — the most powerful vision-language model in the Qwen series to date.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 235
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-235B-A22B-Thinking
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Thinking>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 235
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-235B-A22B-Thinking-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format fp8 --quantization ${quantization}


Model Spec 3 (awq, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 235
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-VL-235B-A22B-Thinking-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format awq --quantization ${quantization}


Model Spec 4 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-30B-A3B-Thinking
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 5 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-30B-A3B-Thinking-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 6 (awq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 8bit
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-VL-30B-A3B-Thinking-AWQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-VL-30B-A3B-Thinking-AWQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-VL-30B-A3B-Thinking-AWQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format awq --quantization ${quantization}


Model Spec 7 (mlx, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 235
- **Quantizations:** 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-235B-A22B-Thinking-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-235B-A22B-Thinking-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-235B-A22B-Thinking-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format mlx --quantization ${quantization}


Model Spec 8 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-30B-A3B-Thinking-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Thinking-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-30B-A3B-Thinking-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 9 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-2B-Thinking-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-2B-Thinking-4bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-2B-Thinking-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 10 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-2B-Thinking-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-2B-Thinking-8bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-2B-Thinking-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 11 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-2B-Thinking-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-2B-Thinking-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-2B-Thinking-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 12 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Thinking-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Thinking-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Thinking-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 13 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Thinking-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Thinking-4bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Thinking-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 14 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Thinking-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Thinking-8bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Thinking-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 15 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Thinking-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Thinking-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Thinking-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 16 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Thinking-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Thinking-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Thinking-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 17 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Thinking-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Thinking-4bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Thinking-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 18 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 5bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Thinking-5bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Thinking-5bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Thinking-5bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 19 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Thinking-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Thinking-8bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Thinking-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 20 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Thinking-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Thinking-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Thinking-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 21 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-30B-A3B-Thinking-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Thinking-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-30B-A3B-Thinking-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 22 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-30B-A3B-Thinking-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Thinking-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-30B-A3B-Thinking-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 23 (mlx, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 235
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-235B-A22B-Thinking-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-235B-A22B-Thinking-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-235B-A22B-Thinking-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format mlx --quantization ${quantization}

