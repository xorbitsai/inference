.. _models_llm_starcoder:

========================================
starcoder
========================================

- **Context Length:** 8192
- **Model Name:** starcoder
- **Languages:** en
- **Abilities:** generate
- **Description:** Starcoder is an open-source Transformer based LLM that is trained on permissively licensed data from GitHub.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggmlv3, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 16
- **Quantizations:** q4_0, q4_1, q5_0, q5_1, q8_0
- **Model ID:** TheBloke/starcoder-GGML
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/starcoder-GGML>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name starcoder --size-in-billions 16 --model-format ggmlv3 --quantization ${quantization}

