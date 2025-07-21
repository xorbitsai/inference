.. _models_llm_glm-4.1v-thinking:

========================================
glm-4.1v-thinking
========================================

- **Context Length:** 65536
- **Model Name:** glm-4.1v-thinking
- **Languages:** en, zh
- **Abilities:** chat, vision, reasoning
- **Description:** GLM-4.1V-9B-Thinking, designed to explore the upper limits of reasoning in vision-language models.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** THUDM/GLM-4.1V-9B-Thinking
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/GLM-4.1V-9B-Thinking>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-4.1V-9B-Thinking>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.1v-thinking --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 9
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** dengcao/GLM-4.1V-9B-Thinking-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/dengcao/GLM-4.1V-9B-Thinking-AWQ>`__, `ModelScope <https://modelscope.cn/models/dengcao/GLM-4.1V-9B-Thinking-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.1v-thinking --size-in-billions 9 --model-format awq --quantization ${quantization}


Model Spec 3 (gptq, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 9
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers
- **Model ID:** dengcao/GLM-4.1V-9B-Thinking-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/dengcao/GLM-4.1V-9B-Thinking-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/dengcao/GLM-4.1V-9B-Thinking-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-4.1v-thinking --size-in-billions 9 --model-format gptq --quantization ${quantization}

