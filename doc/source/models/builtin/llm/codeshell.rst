.. _models_llm_codeshell:

========================================
codeshell
========================================

- **Context Length:** 8194
- **Model Name:** codeshell
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** CodeShell is a multi-language code LLM developed by the Knowledge Computing Lab of Peking University. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** WisdomShell/CodeShell-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/WisdomShell/CodeShell-7B>`__, `ModelScope <https://modelscope.cn/models/WisdomShell/CodeShell-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name codeshell --size-in-billions 7 --model-format pytorch --quantization ${quantization}

