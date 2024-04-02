.. _models_llm_aquila2:

========================================
aquila2
========================================

- **Context Length:** 2048
- **Model Name:** aquila2
- **Languages:** zh
- **Abilities:** generate
- **Description:** Aquila2 series models are the base language models

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Model ID:** BAAI/Aquila2-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/Aquila2-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name aquila2 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** none
- **Model ID:** BAAI/Aquila2-34B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/Aquila2-34B>`__, `ModelScope <https://modelscope.cn/models/BAAI/Aquila2-34B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name aquila2 --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Model ID:** BAAI/Aquila2-70B-Expr
- **Model Hubs**:  `Hugging Face <https://huggingface.co/BAAI/Aquila2-70B-Expr>`__, `ModelScope <https://modelscope.cn/models/BAAI/Aquila2-70B-Expr>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name aquila2 --size-in-billions 70 --model-format pytorch --quantization ${quantization}

