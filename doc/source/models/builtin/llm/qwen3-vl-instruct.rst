.. _models_llm_qwen3-vl-instruct:

========================================
Qwen3-VL-Instruct
========================================

- **Context Length:** 262144
- **Model Name:** Qwen3-VL-Instruct
- **Languages:** en, zh
- **Abilities:** chat, vision, tools
- **Description:** Meet Qwen3-VL — the most powerful vision-language model in the Qwen series to date.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 235
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-235B-A22B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 235 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 235
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 235 --model-format fp8 --quantization ${quantization}


Model Spec 3 (awq, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 235
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-VL-235B-A22B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 235 --model-format awq --quantization ${quantization}


Model Spec 4 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-30B-A3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 5 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 6 (awq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 8bit
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-VL-30B-A3B-Instruct-AWQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-VL-30B-A3B-Instruct-AWQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 30 --model-format awq --quantization ${quantization}


Model Spec 7 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-32B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 8 (fp8, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 32
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-32B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 32 --model-format fp8 --quantization ${quantization}


Model Spec 9 (awq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 32
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-VL-32B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-VL-32B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-VL-32B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 32 --model-format awq --quantization ${quantization}


Model Spec 10 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-8B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 11 (fp8, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 8
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-8B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 8 --model-format fp8 --quantization ${quantization}


Model Spec 12 (awq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 8
- **Quantizations:** 4bit, 8bit
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-VL-8B-Instruct-AWQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-VL-8B-Instruct-AWQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-VL-8B-Instruct-AWQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 8 --model-format awq --quantization ${quantization}


Model Spec 13 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-4B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 14 (fp8, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 4
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-4B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 4 --model-format fp8 --quantization ${quantization}


Model Spec 15 (awq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 4
- **Quantizations:** 4bit, 8bit
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-VL-4B-Instruct-AWQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-VL-4B-Instruct-AWQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-VL-4B-Instruct-AWQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 4 --model-format awq --quantization ${quantization}


Model Spec 16 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-2B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 17 (fp8, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 2
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-2B-Instruct-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 2 --model-format fp8 --quantization ${quantization}


Model Spec 18 (mlx, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 235
- **Quantizations:** 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-235B-A22B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-235B-A22B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-235B-A22B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 235 --model-format mlx --quantization ${quantization}


Model Spec 19 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-30B-A3B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-30B-A3B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 20 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-32B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-32B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-32B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 32 --model-format mlx --quantization ${quantization}


Model Spec 21 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 22 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** 4bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-2B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-2B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 23 (mlx, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 2
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-2B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-2B-Instruct-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 2 --model-format mlx --quantization ${quantization}


Model Spec 24 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Instruct-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Instruct-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 25 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Instruct-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Instruct-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 26 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Instruct-8bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-8bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Instruct-8bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 27 (mlx, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 4
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-4B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-4B-Instruct-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 4 --model-format mlx --quantization ${quantization}


Model Spec 28 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Instruct-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Instruct-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 29 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 5bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Instruct-5bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-5bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Instruct-5bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 30 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-8B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-8B-Instruct-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 8 --model-format mlx --quantization ${quantization}


Model Spec 31 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 32 (mlx, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 30
- **Quantizations:** bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 30 --model-format mlx --quantization ${quantization}


Model Spec 33 (mlx, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 32
- **Quantizations:** 3bit
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen3-VL-32B-Instruct-3bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen3-VL-32B-Instruct-3bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen3-VL-32B-Instruct-3bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Instruct --size-in-billions 32 --model-format mlx --quantization ${quantization}

