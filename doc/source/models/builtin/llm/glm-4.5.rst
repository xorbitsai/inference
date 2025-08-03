.. _models_llm_glm-4.5:

========================================
glm-4.5
========================================

- **Context Length:** 65536
- **Model Name:** glm-4.5
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid
- **Description:** The GLM-4.5 series models are foundation models designed for intelligent agents. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 355
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** zai-org/GLM-4.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.5>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 355 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 355
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** zai-org/GLM-4.5-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.5-FP8>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.5-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 355 --model-format fp8 --quantization ${quantization}


Model Spec 3 (mlx, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 355
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** mlx-community/GLM-4.5-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/GLM-4.5-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/GLM-4.5-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 355 --model-format mlx --quantization ${quantization}


Model Spec 4 (pytorch, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 106
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** zai-org/GLM-4.5-Air
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.5-Air>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.5-Air>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 106 --model-format pytorch --quantization ${quantization}


Model Spec 5 (fp8, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 106
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** zai-org/GLM-4.5-Air-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.5-Air-FP8>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.5-Air-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 106 --model-format fp8 --quantization ${quantization}


Model Spec 6 (mlx, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 106
- **Quantizations:** 2bit, 3bit, 4bit, 5bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/GLM-4.5-Air-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/GLM-4.5-Air-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/GLM-4.5-Air-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 106 --model-format mlx --quantization ${quantization}

