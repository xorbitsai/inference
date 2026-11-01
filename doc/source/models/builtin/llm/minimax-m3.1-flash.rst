.. _models_llm_minimax-m3.1-flash:

========================================
MiniMax-M3.1-Flash
========================================

- **Context Length:** 1048576
- **Model Name:** MiniMax-M3.1-Flash
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning
- **Description:** MiniMax-M3.1-Flash is a native multimodal Mixture-of-Experts model with 428B total parameters, 23B activated parameters and a 1M-token context window. It uses sparse attention throughout and quantization-aware training with MXFP8 dense layers and NVFP4 experts.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (fp8, 428 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 428
- **Quantizations:** mxfp8
- **Engines**: vLLM
- **Model ID:** MiniMaxAI/MiniMax-M3.1-Flash
- **Model Hubs**:  `Hugging Face <https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name MiniMax-M3.1-Flash --size-in-billions 428 --model-format fp8 --quantization ${quantization}
