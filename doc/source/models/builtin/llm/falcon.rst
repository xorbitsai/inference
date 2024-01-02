.. _models_llm_falcon:

========================================
falcon
========================================

- **Context Length:** 2048
- **Model Name:** falcon
- **Languages:** en
- **Abilities:** generate
- **Description:** Falcon is an open-source Transformer based LLM trained on the RefinedWeb dataset.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 40 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 40
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** tiiuae/falcon-40b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tiiuae/falcon-40b>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name falcon --size-in-billions 40 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** tiiuae/falcon-7b
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tiiuae/falcon-7b>`_

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name falcon --size-in-billions 7 --model-format pytorch --quantization ${quantization}

