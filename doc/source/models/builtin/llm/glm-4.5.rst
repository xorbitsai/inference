.. _models_llm_glm-4.5:

========================================
glm-4.5
========================================

- **Context Length:** 131072
- **Model Name:** glm-4.5
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
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


Model Spec 3 (gptq, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 355
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/GLM-4.5-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/GLM-4.5-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/GLM-4.5-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 355 --model-format gptq --quantization ${quantization}


Model Spec 4 (awq, 355 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 355
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/GLM-4.5-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/GLM-4.5-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/GLM-4.5-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 355 --model-format awq --quantization ${quantization}


Model Spec 5 (mlx, 355 Billion)
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


Model Spec 6 (pytorch, 106 Billion)
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


Model Spec 7 (fp8, 106 Billion)
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


Model Spec 8 (gptq, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 106
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/GLM-4.5-Air-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/GLM-4.5-Air-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/GLM-4.5-Air-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 106 --model-format gptq --quantization ${quantization}


Model Spec 9 (awq, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 106
- **Quantizations:** AWQ-FP16Mix
- **Engines**: Transformers
- **Model ID:** QuantTrio/GLM-4.5-Air-AWQ-FP16Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/GLM-4.5-Air-AWQ-FP16Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/GLM-4.5-Air-AWQ-FP16Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 106 --model-format awq --quantization ${quantization}


Model Spec 10 (awq, 106 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 106
- **Quantizations:** 4bit
- **Engines**: 
- **Model ID:** cpatonn-mirror/GLM-4.5-Air-AWQ-4bit
- **Model Hubs**:  `ModelScope <https://modelscope.cn/models/cpatonn-mirror/GLM-4.5-Air-AWQ-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.5 --size-in-billions 106 --model-format awq --quantization ${quantization}


Model Spec 11 (mlx, 106 Billion)
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

