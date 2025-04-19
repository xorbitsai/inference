.. _models_llm_seallms-v3:

========================================
seallms-v3
========================================

- **Context Length:** 32768
- **Model Name:** seallms-v3
- **Languages:** en, zh, id, vi, th, ph, ms, mm, kh, la, in
- **Abilities:** chat
- **Description:** SeaLLMs - Large Language Models for Southeast Asia

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** SeaLLMs/SeaLLMs-v3-1.5B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/SeaLLMs/SeaLLMs-v3-1.5B-Chat>`__, `ModelScope <https://modelscope.cn/models/SeaLLMs/SeaLLMs-v3-1.5B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seallms-v3 --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** SeaLLMs/SeaLLMs-v3-7B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat>`__, `ModelScope <https://modelscope.cn/models/SeaLLMs/SeaLLMs-v3-7B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seallms-v3 --size-in-billions 7 --model-format pytorch --quantization ${quantization}

