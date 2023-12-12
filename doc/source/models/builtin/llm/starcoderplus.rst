.. _models_llm_starcoderplus:

========================================
starcoderplus
========================================

- **Context Length:** 8192
- **Model Name:** starcoderplus
- **Languages:** en
- **Abilities:** generate
- **Description:** Starcoderplus is an open-source LLM trained by fine-tuning Starcoder on RedefinedWeb and StarCoderData datasets.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** bigcode/starcoderplus

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name starcoderplus --size-in-billions 16 --model-format pytorch --quantization ${quantization}

