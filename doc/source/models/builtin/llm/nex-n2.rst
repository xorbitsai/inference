.. _models_llm_nex-n2:

========================================
Nex-N2
========================================

- **Context Length:** 262144
- **Model Name:** Nex-N2
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning, hybrid
- **Description:** Nex-N2 is a series of multimodal large language models developed by nex-agi, built on the Qwen3.5 MoE architecture. It supports text, image, and video understanding, with advanced reasoning and tool-use capabilities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 35 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 35
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** nex-agi/Nex-N2-mini
- **Model Hubs**:  `Hugging Face <https://huggingface.co/nex-agi/Nex-N2-mini>`__, `ModelScope <https://modelscope.cn/models/nex-agi/Nex-N2-mini>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Nex-N2 --size-in-billions 35 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 397
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** nex-agi/Nex-N2-Pro
- **Model Hubs**:  `Hugging Face <https://huggingface.co/nex-agi/Nex-N2-Pro>`__, `ModelScope <https://modelscope.cn/models/nex-agi/Nex-N2-Pro>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Nex-N2 --size-in-billions 397 --model-format pytorch --quantization ${quantization}


Model Spec 3 (fp8, 397 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 397
- **Quantizations:** FP8
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** nex-agi/Nex-N2-Pro-fp8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/nex-agi/Nex-N2-Pro-fp8>`__, `ModelScope <https://modelscope.cn/models/nex-agi/Nex-N2-Pro-fp8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Nex-N2 --size-in-billions 397 --model-format fp8 --quantization ${quantization}

