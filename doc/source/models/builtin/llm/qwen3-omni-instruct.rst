.. _models_llm_qwen3-omni-instruct:

========================================
Qwen3-Omni-Instruct
========================================

- **Context Length:** 262144
- **Model Name:** Qwen3-Omni-Instruct
- **Languages:** en, zh
- **Abilities:** chat, vision, audio, omni, tools
- **Description:** Qwen3-Omni is the natively end-to-end multilingual omni-modal foundation models. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. We introduce several architectural upgrades to improve performance and efficiency.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-Omni-30B-A3B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Omni-Instruct --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 8bit
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-Omni-30B-A3B-Instruct-AWQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-Omni-Instruct --size-in-billions 30 --model-format awq --quantization ${quantization}

