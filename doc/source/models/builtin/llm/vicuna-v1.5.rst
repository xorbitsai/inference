.. _models_llm_vicuna-v1.5:

========================================
vicuna-v1.5
========================================

- **Context Length:** 4096
- **Model Name:** vicuna-v1.5
- **Languages:** en
- **Abilities:** chat
- **Description:** Vicuna is an open-source LLM trained by fine-tuning LLaMA on data collected from ShareGPT.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** lmsys/vicuna-7b-v1.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmsys/vicuna-7b-v1.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name vicuna-v1.5 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 13 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 13
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** lmsys/vicuna-13b-v1.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmsys/vicuna-13b-v1.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name vicuna-v1.5 --size-in-billions 13 --model-format pytorch --quantization ${quantization}

