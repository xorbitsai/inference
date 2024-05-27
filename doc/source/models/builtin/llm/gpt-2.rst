.. _models_llm_gpt-2:

========================================
gpt-2
========================================

- **Context Length:** 1024
- **Model Name:** gpt-2
- **Languages:** en
- **Abilities:** generate
- **Description:** GPT-2 is a Transformer-based LLM that is trained on WebTest, a 40 GB dataset of Reddit posts with 3+ upvotes.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_5 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_5
- **Quantizations:** none
- **Engines**: 
- **Model ID:** openai-community/gpt2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/openai-community/gpt2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gpt-2 --size-in-billions 1_5 --model-format pytorch --quantization ${quantization}

