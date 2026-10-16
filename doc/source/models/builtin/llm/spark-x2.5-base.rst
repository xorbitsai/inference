.. _models_llm_spark-x2.5-base:

========================================
Spark-X2.5-Base
========================================

- **Context Length:** 1048576
- **Model Name:** Spark-X2.5-Base
- **Languages:** en, zh
- **Abilities:** generate
- **Description:** Spark-X2.5 base checkpoints in 1.7B and 4B sizes for continued pretraining and completion workloads.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1_7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1_7
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** XHToken/Spark-X2.5-1.7B-Base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-1.7B-Base>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-1.7B-Base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5-Base --size-in-billions 1_7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** XHToken/Spark-X2.5-4B-Base
- **Model Hubs**:  `Hugging Face <https://huggingface.co/XHToken/Spark-X2.5-4B-Base>`__, `ModelScope <https://modelscope.cn/models/XHToken/Spark-X2.5-4B-Base>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Spark-X2.5-Base --size-in-billions 4 --model-format pytorch --quantization ${quantization}
