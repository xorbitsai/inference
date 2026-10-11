.. _models_llm_glm-5.3:

========================================
glm-5.3
========================================

- **Context Length:** 1048576
- **Model Name:** glm-5.3
- **Languages:** en, zh
- **Abilities:** chat, tools, reasoning
- **Description:** GLM-5.3 is a reasoning model for coding and long-horizon agentic tasks.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 753 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 753
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** zai-org/GLM-5.3-BF16
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-5.3-BF16>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-5.3-BF16>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.3 --size-in-billions 753 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 753 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 753
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** zai-org/GLM-5.3
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-5.3>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-5.3>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5.3 --size-in-billions 753 --model-format fp8 --quantization ${quantization}

