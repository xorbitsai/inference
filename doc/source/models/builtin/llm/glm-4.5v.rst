.. _models_llm_glm-4.5v:

========================================
glm-4.5v
========================================

- **Context Length:** 131072
- **Model Name:** glm-4.5v
- **Languages:** en, zh
- **Abilities:** chat, vision, reasoning
- **Description:** GLM-4.5V is based on ZhipuAI’s next-generation flagship text foundation model GLM-4.5-Air (106B parameters, 12B active). It continues the technical approach of GLM-4.1V-Thinking, achieving SOTA performance among models of the same scale on 42 public vision-language benchmarks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 106
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** zai-org/GLM-4.5V
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.5V>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.5V>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5v --size-in-billions 106 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 106
- **Quantizations:** FP8
- **Engines**: vLLM, Transformers
- **Model ID:** zai-org/GLM-4.5V-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.5V-FP8>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.5V-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5v --size-in-billions 106 --model-format fp8 --quantization ${quantization}


Model Spec 3 (awq, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 106
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/GLM-4.5V-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/GLM-4.5V-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/GLM-4.5V-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5v --size-in-billions 106 --model-format awq --quantization ${quantization}


Model Spec 4 (mlx, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 106
- **Quantizations:** 3bit, 4bit, 5bit, 6bit, 8bit
- **Engines**: Transformers, MLX
- **Model ID:** mlx-community/GLM-4.5V-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/GLM-4.5V-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/GLM-4.5V-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5v --size-in-billions 106 --model-format mlx --quantization ${quantization}

