.. _models_llm_skywork-or1:

========================================
skywork-or1
========================================

- **Context Length:** 131072
- **Model Name:** skywork-or1
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** We release the final version of Skywork-OR1 (Open Reasoner 1) series of models, including

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 32
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Skywork/Skywork-OR1-32B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Skywork/Skywork-OR1-32B>`__, `ModelScope <https://modelscope.cn/models/Skywork/Skywork-OR1-32B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1 --size-in-billions 32 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 32 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 32
- **Quantizations:** Int8, Int4
- **Engines**: Transformers
- **Model ID:** JunHowie/Skywork-OR1-32B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Skywork-OR1-32B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Skywork-OR1-32B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1 --size-in-billions 32 --model-format gptq --quantization ${quantization}


Model Spec 3 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Skywork/Skywork-OR1-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Skywork/Skywork-OR1-7B>`__, `ModelScope <https://modelscope.cn/models/Skywork/Skywork-OR1-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 4 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int8, Int4
- **Engines**: Transformers
- **Model ID:** JunHowie/Skywork-OR1-7B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/JunHowie/Skywork-OR1-7B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/JunHowie/Skywork-OR1-7B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name skywork-or1 --size-in-billions 7 --model-format gptq --quantization ${quantization}

