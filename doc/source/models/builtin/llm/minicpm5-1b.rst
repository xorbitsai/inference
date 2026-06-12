.. _models_llm_minicpm5-1b:

========================================
minicpm5-1b
========================================

- **Context Length:** 131072
- **Model Name:** minicpm5-1b
- **Languages:** en, zh
- **Abilities:** chat, reasoning, hybrid, tools
- **Description:** MiniCPM5-1B is the first model in the MiniCPM5 series. It is a dense 1B Transformer built for on-device, local deployment, and resource-constrained scenarios, reaching 1B-class open-source SOTA. Supports hybrid thinking via enable_thinking and native XML-style tool calling (MCP-compatible).

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** openbmb/MiniCPM5-1B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM5-1B>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM5-1B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm5-1b --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 1
- **Quantizations:** F16, Q4_K_M, Q8_0
- **Engines**: vLLM, llama.cpp
- **Model ID:** openbmb/MiniCPM5-1B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM5-1B-GGUF>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM5-1B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm5-1b --size-in-billions 1 --model-format ggufv2 --quantization ${quantization}


Model Spec 3 (mlx, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 1
- **Quantizations:** 4bit
- **Engines**: MLX
- **Model ID:** openbmb/MiniCPM5-1B-MLX
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openbmb/MiniCPM5-1B-MLX>`__, `ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM5-1B-MLX>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name minicpm5-1b --size-in-billions 1 --model-format mlx --quantization ${quantization}

