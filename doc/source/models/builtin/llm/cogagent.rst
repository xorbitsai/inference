.. _models_llm_cogagent:

========================================
cogagent
========================================

- **Context Length:** 4096
- **Model Name:** cogagent
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** The CogAgent-9B-20241220 model is based on GLM-4V-9B, a bilingual open-source VLM base model. Through data collection and optimization, multi-stage training, and strategy improvements, CogAgent-9B-20241220 achieves significant advancements in GUI perception, inference prediction accuracy, action space completeness, and task generalizability. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** THUDM/cogagent-9b-20241220
- **Model Hubs**:  `Hugging Face <https://huggingface.co/THUDM/cogagent-9b-20241220>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/cogagent-9b-20241220>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name cogagent --size-in-billions 9 --model-format pytorch --quantization ${quantization}

