.. _models_llm_qwen2.5-vl-instruct:

========================================
qwen2.5-vl-instruct
========================================

- **Context Length:** 128000
- **Model Name:** qwen2.5-vl-instruct
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** Qwen2.5-VL: Qwen2.5-VL is the latest version of the vision language models in the Qwen model familities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen2.5-VL-3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen2.5-VL-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-VL-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen2.5-VL-72B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-VL-72B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 4 (mlx, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 3
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-VL-3B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-VL-3B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen2.5-VL-3B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 3 --model-format mlx --quantization ${quantization}


Model Spec 5 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-VL-7B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-VL-7B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen2.5-VL-7B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 7 --model-format mlx --quantization ${quantization}


Model Spec 6 (mlx, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 72
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/Qwen2.5-VL-72B-Instruct-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/Qwen2.5-VL-72B-Instruct-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/Qwen2.5-VL-72B-Instruct-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 72 --model-format mlx --quantization ${quantization}

