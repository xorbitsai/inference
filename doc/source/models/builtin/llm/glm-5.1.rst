.. _models_llm_glm-5.1:

========================================
glm-5.1
========================================

- **Context Length:** 202752
- **Model Name:** glm-5.1
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** GLM-5.1 is our next-generation flagship model for agentic engineering, with significantly stronger coding capabilities than its predecessor. It achieves state-of-the-art performance on SWE-Bench Pro and leads GLM-5 by a wide margin on NL2Repo (repo generation) and Terminal-Bench 2.0 (real-world terminal tasks).

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 744 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 744
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** zai-org/GLM-5.1
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-5.1>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-5.1>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.1 --size-in-billions 744 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 744 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 744
- **Quantizations:** FP8
- **Engines**: 
- **Model ID:** zai-org/GLM-5.1-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-5.1-FP8>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-5.1-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.1 --size-in-billions 744 --model-format fp8 --quantization ${quantization}


Model Spec 3 (ggufv2, 744 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 744
- **Quantizations:** none
- **Engines**: 
- **Model ID:** unsloth/GLM-5.1-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/unsloth/GLM-5.1-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/GLM-5.1-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.1 --size-in-billions 744 --model-format ggufv2 --quantization ${quantization}


Model Spec 4 (mlx, 744 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 744
- **Quantizations:** 8bit-MXFP8
- **Engines**: 
- **Model ID:** mlx-community/GLM-5.1-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/GLM-5.1-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/GLM-5.1-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.1 --size-in-billions 744 --model-format mlx --quantization ${quantization}

