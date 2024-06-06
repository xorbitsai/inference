.. _models_llm_internvl-chat:

========================================
internvl-chat
========================================

- **Context Length:** 32768
- **Model Name:** internvl-chat
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** InternVL 1.5 is an open-source multimodal large language model (MLLM) to bridge the capability gap between open-source and proprietary commercial models in multimodal understanding. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** OpenGVLab/Mini-InternVL-Chat-2B-V1-5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl-chat --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 26
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** OpenGVLab/InternVL-Chat-V1-5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/InternVL-Chat-V1-5-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl-chat --size-in-billions 26 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 26
- **Quantizations:** Int8
- **Engines**: Transformers
- **Model ID:** OpenGVLab/InternVL-Chat-V1-5-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-{quantization}>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/InternVL-Chat-V1-5-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl-chat --size-in-billions 26 --model-format pytorch --quantization ${quantization}

