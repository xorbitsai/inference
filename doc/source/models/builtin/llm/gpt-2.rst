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


Model Spec 1 (ggmlv3, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Model ID:** marella/gpt-2-ggml

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name gpt-2 --size-in-billions 1 --model-format ggmlv3 --quantization ${quantization}

