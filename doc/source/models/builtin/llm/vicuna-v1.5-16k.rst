.. _models_llm_vicuna-v1.5-16k:

========================================
vicuna-v1.5-16k
========================================

- **Context Length:** 16384
- **Model Name:** vicuna-v1.5-16k
- **Languages:** en
- **Abilities:** chat
- **Description:** Vicuna-v1.5-16k is a special version of Vicuna-v1.5, with a context window of 16k tokens instead of 4k.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: 
- **Model ID:** lmsys/vicuna-7b-v1.5-16k
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmsys/vicuna-7b-v1.5-16k>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name vicuna-v1.5-16k --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: 
- **Model ID:** lmsys/vicuna-13b-v1.5-16k
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmsys/vicuna-13b-v1.5-16k>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name vicuna-v1.5-16k --size-in-billions 13 --model-format pytorch --quantization ${quantization}

