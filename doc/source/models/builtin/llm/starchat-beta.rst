.. _models_llm_starchat-beta:

========================================
starchat-beta
========================================

- **Context Length:** 8192
- **Model Name:** starchat-beta
- **Languages:** en
- **Abilities:** chat
- **Description:** Starchat-beta is a fine-tuned version of the Starcoderplus LLM, specializing in coding assistance.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** HuggingFaceH4/starchat-beta
- **Model Hubs**:  `Hugging Face <https://huggingface.co/HuggingFaceH4/starchat-beta>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name starchat-beta --size-in-billions 16 --model-format pytorch --quantization ${quantization}

