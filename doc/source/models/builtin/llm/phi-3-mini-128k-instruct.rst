.. _models_llm_phi-3-mini-128k-instruct:

========================================
phi-3-mini-128k-instruct
========================================

- **Context Length:** 128000
- **Model Name:** phi-3-mini-128k-instruct
- **Languages:** en
- **Abilities:** chat
- **Description:** The Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open model trained using the Phi-3 datasets.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** microsoft/Phi-3-mini-128k-instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/microsoft/Phi-3-mini-128k-instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Phi-3-mini-128k-instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name phi-3-mini-128k-instruct --size-in-billions 4 --model-format pytorch --quantization ${quantization}

