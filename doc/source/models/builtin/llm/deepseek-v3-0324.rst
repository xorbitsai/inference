.. _models_llm_deepseek-v3-0324:

========================================
deepseek-v3-0324
========================================

- **Context Length:** 163840
- **Model Name:** deepseek-v3-0324
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 671
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/DeepSeek-V3-0324
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-V3-0324>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-V3-0324>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v3-0324 --size-in-billions 671 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 671
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** cognitivecomputations/DeepSeek-V3-0324-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cognitivecomputations/DeepSeek-V3-0324-AWQ>`__, `ModelScope <https://modelscope.cn/models/cognitivecomputations/DeepSeek-V3-0324-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-v3-0324 --size-in-billions 671 --model-format awq --quantization ${quantization}

