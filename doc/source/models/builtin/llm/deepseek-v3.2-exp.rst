.. _models_llm_deepseek-v3.2-exp:

========================================
DeepSeek-V3.2-Exp
========================================

- **Context Length:** 163840
- **Model Name:** DeepSeek-V3.2-Exp
- **Languages:** en, zh
- **Abilities:** chat, reasoning
- **Description:** We are excited to announce the official release of DeepSeek-V3.2-Exp, an experimental version of our model. As an intermediate step toward our next-generation architecture, V3.2-Exp builds upon V3.1-Terminus by introducing DeepSeek Sparse Attention—a sparse attention mechanism designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 671
- **Quantizations:** none
- **Engines**: 
- **Model ID:** deepseek-ai/DeepSeek-V3.2-Exp
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.2-Exp>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name DeepSeek-V3.2-Exp --size-in-billions 671 --model-format pytorch --quantization ${quantization}

