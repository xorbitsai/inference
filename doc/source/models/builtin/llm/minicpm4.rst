.. _models_llm_minicpm4:

========================================
minicpm4
========================================

- **Context Length:** 32768
- **Model Name:** minicpm4
- **Languages:** zh
- **Abilities:** chat
- **Description:** MiniCPM4 series are highly efficient large language models (LLMs) designed explicitly for end-side devices, which achieves this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 0_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_5
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM4-0.5B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM4-0.5B>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM4-0.5B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm4 --size-in-billions 0_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** openbmb/MiniCPM4-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM4-8B>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM4-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm4 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 3 (mlx, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 8
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/MiniCPM4-8B-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/MiniCPM4-8B-4bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/MiniCPM4-8B-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm4 --size-in-billions 8 --model-format mlx --quantization ${quantization}

