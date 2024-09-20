.. _models_llm_qwen2-audio-instruct:

========================================
qwen2-audio-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2-audio-instruct
- **Languages:** en, zh
- **Abilities:** chat, audio
- **Description:** Qwen2-Audio: A large-scale audio-language model which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-Audio-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-Audio-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-audio-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}

