.. _models_llm_glm-4.7:

========================================
GLM-4.7
========================================

- **Context Length:** 202752
- **Model Name:** GLM-4.7
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** GLM-4.7 significantly advances core and multilingual agentic coding, UI/vibe coding, tool use, and complex reasoning—outperforming GLM-4.6 across benchmarks like SWE-bench, Terminal Bench 2.0, τ²-Bench, and HLE—while also improving chat, creative writing, and role-play.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 355
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** zai-org/GLM-4.7
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.7>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.7>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name GLM-4.7 --size-in-billions 355 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 355
- **Quantizations:** FP8
- **Engines**: vLLM
- **Model ID:** zai-org/GLM-4.7-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-4.7-FP8>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.7-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name GLM-4.7 --size-in-billions 355 --model-format fp8 --quantization ${quantization}


Model Spec 3 (gptq, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 355
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/GLM-4.7-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/GLM-4.7-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/GLM-4.7-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name GLM-4.7 --size-in-billions 355 --model-format gptq --quantization ${quantization}


Model Spec 4 (awq, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 355
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/GLM-4.7-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/GLM-4.7-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/GLM-4.7-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name GLM-4.7 --size-in-billions 355 --model-format awq --quantization ${quantization}


Model Spec 5 (mlx, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 355
- **Quantizations:** 4bit, 6bit, 8bit
- **Engines**: MLX
- **Model ID:** mlx-community/GLM-4.7-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/GLM-4.7-{quantization}>`__, `ModelScope <https://modelscope.cn/models/mlx-community/GLM-4.7-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name GLM-4.7 --size-in-billions 355 --model-format mlx --quantization ${quantization}

