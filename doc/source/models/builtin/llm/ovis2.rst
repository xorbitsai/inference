.. _models_llm_ovis2:

========================================
Ovis2
========================================

- **Context Length:** 32768
- **Model Name:** Ovis2
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** Ovis (Open VISion) is a novel Multimodal Large Language Model (MLLM) architecture, designed to structurally align visual and textual embeddings.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-1B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-1B>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-1B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-2B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-2B>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-2B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-4B>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 4 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-8B>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 5 (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-16B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-16B>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-16B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 16 --model-format pytorch --quantization ${quantization}


Model Spec 6 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-34B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-34B>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-34B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 7 (gptq, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 2
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-2B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-2B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-2B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 2 --model-format gptq --quantization ${quantization}


Model Spec 8 (gptq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 4
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-4B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-4B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-4B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 4 --model-format gptq --quantization ${quantization}


Model Spec 9 (gptq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-8B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-8B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-8B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 8 --model-format gptq --quantization ${quantization}


Model Spec 10 (gptq, 16 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 16
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-16B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-16B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-16B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 16 --model-format gptq --quantization ${quantization}


Model Spec 11 (gptq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 34
- **Quantizations:** Int4, Int8
- **Engines**: Transformers
- **Model ID:** AIDC-AI/Ovis2-34B-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/AIDC-AI/Ovis2-34B-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/AIDC-AI/Ovis2-34B-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Ovis2 --size-in-billions 34 --model-format gptq --quantization ${quantization}

