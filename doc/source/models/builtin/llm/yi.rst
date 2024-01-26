.. _models_llm_yi:

========================================
Yi
========================================

- **Context Length:** 4096
- **Model Name:** Yi
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** The Yi series models are large language models trained from scratch by developers at 01.AI.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q3_K_M, Q3_K_S, Q4_0, Q4_K_M, Q4_K_S, Q5_0, Q5_K_M, Q5_K_S, Q6_K, Q8_0
- **Model ID:** TheBloke/Yi-34B-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/Yi-34B-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** 01-ai/Yi-6B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-6B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-6B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi --size-in-billions 6 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** 01-ai/Yi-34B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-34B>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-34B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name Yi --size-in-billions 34 --model-format pytorch --quantization ${quantization}

