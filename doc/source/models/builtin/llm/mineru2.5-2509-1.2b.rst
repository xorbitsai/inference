.. _models_llm_mineru2.5-2509-1.2b:

========================================
MinerU2.5-2509-1.2B
========================================

- **Context Length:** 32768
- **Model Name:** MinerU2.5-2509-1.2B
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** MinerU2.5-2509-1.2B is a vision language model for document understanding.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_2
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** opendatalab/MinerU2.5-2509-1.2B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B>`__, `ModelScope <https://modelscope.cn/models/opendatalab/MinerU2.5-2509-1.2B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MinerU2.5-2509-1.2B --size-in-billions 1_2 --model-format pytorch --quantization ${quantization}

