.. _models_llm_qwen2.5-vl-instruct:

========================================
qwen2.5-vl-instruct
========================================

- **Context Length:** 128000
- **Model Name:** qwen2.5-vl-instruct
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** Qwen2.5-VL: Qwen2.5-VL is the latest version of the vision language models in the Qwen model familities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2.5-VL-3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-VL-3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 3 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2.5-VL-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-VL-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2.5-VL-72B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2.5-VL-72B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2.5-vl-instruct --size-in-billions 72 --model-format pytorch --quantization ${quantization}

