.. _models_llm_gemma-it:

========================================
gemma-it
========================================

- **Context Length:** 8192
- **Model Name:** gemma-it
- **Languages:** en
- **Abilities:** chat
- **Description:** Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none, 4-bit, 8-bit
- **Engines**: Transformers, vLLM (vLLM only available for quantization none)
- **Model ID:** google/gemma-2b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-2b-it>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/gemma-2b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-it --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none, 4-bit, 8-bit
- **Engines**: Transformers, vLLM (vLLM only available for quantization none)
- **Model ID:** google/gemma-7b-it
- **Model Hubs**:  `Hugging Face <https://huggingface.co/google/gemma-7b-it>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/gemma-7b-it>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name gemma-it --size-in-billions 7 --model-format pytorch --quantization ${quantization}

