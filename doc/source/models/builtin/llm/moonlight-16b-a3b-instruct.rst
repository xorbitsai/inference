.. _models_llm_moonlight-16b-a3b-instruct:

========================================
moonlight-16b-a3b-instruct
========================================

- **Context Length:** 8192
- **Model Name:** moonlight-16b-a3b-instruct
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Kimi Muon is Scalable for LLM Training

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 3 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** moonshotai/Moonlight-16B-A3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/moonshotai/Moonlight-16B-A3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name moonlight-16b-a3b-instruct --size-in-billions 3 --model-format pytorch --quantization ${quantization}

