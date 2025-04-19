.. _models_llm_deepseek-vl2:

========================================
deepseek-vl2
========================================

- **Context Length:** 4096
- **Model Name:** deepseek-vl2
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL. DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 27 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 27
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** deepseek-ai/deepseek-vl2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-vl2>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-vl2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-vl2 --size-in-billions 27 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** deepseek-ai/deepseek-vl2-small
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-vl2-small>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-vl2-small>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-vl2 --size-in-billions 16 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** deepseek-ai/deepseek-vl2-tiny
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/deepseek-vl2-tiny>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/deepseek-vl2-tiny>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-vl2 --size-in-billions 3 --model-format pytorch --quantization ${quantization}

