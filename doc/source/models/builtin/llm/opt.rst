.. _models_llm_opt:

========================================
opt
========================================

- **Context Length:** 2048
- **Model Name:** opt
- **Languages:** en
- **Abilities:** generate
- **Description:** Opt is an open-source, decoder-only, Transformer based LLM that was designed to replicate GPT-3.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** facebook/opt-125m
- **Model Hubs**:  `Hugging Face <https://huggingface.co/facebook/opt-125m>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name opt --size-in-billions 1 --model-format pytorch --quantization ${quantization}

