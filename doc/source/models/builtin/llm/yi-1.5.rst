.. _models_llm_yi-1.5:

========================================
Yi-1.5
========================================

- **Context Length:** 4096
- **Model Name:** Yi-1.5
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** 01-ai/Yi-1.5-6B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-6B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-6B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5 --size-in-billions 6 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** 01-ai/Yi-1.5-9B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-9B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-9B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5 --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers
- **Model ID:** 01-ai/Yi-1.5-34B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-34B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-34B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5 --size-in-billions 34 --model-format pytorch --quantization ${quantization}

