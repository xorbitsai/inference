.. _models_builtin_Yi:


==
Yi
==

- **Context Length:** 4096
- **Model Name:** Yi
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** The Yi series models are large language models trained from scratch by developers at 01.AI. The first public release contains two bilingual (English/Chinese) base models with the parameter sizes of 6B and 34B. Both of them are trained with 4K sequence length and can be extended to 32K during inference time.

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 6 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** 01-ai/Yi-6B

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi --size-in-billions 6 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** 01-ai/Yi-34B

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi --size-in-billions 34 --model-format pytorch --quantization ${quantization}


.. note::

   4-bit quantization is not supported on macOS.
