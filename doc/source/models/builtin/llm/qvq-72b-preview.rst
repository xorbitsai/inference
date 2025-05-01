.. _models_llm_qvq-72b-preview:

========================================
QvQ-72B-Preview
========================================

- **Context Length:** 32768
- **Model Name:** QvQ-72B-Preview
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** QVQ-72B-Preview is an experimental research model developed by the Qwen team, focusing on enhancing visual reasoning capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/QVQ-72B-Preview
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/QVQ-72B-Preview>`__, `ModelScope <https://modelscope.cn/models/Qwen/QVQ-72B-Preview>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QvQ-72B-Preview --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 2 (mlx, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 72
- **Quantizations:** 3bit, 4bit, 6bit, 8bit, bf16
- **Engines**: MLX
- **Model ID:** mlx-community/QVQ-72B-Preview-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/QVQ-72B-Preview-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/QVQ-72B-Preview-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name QvQ-72B-Preview --size-in-billions 72 --model-format mlx --quantization ${quantization}

