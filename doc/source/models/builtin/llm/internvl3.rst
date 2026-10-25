.. _models_llm_internvl3:

========================================
InternVL3
========================================

- **Context Length:** 8192
- **Model Name:** InternVL3
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** InternVL3, an advanced multimodal large language model (MLLM) series that demonstrates superior overall performance.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-1B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-1B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-1B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (awq, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 1
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-1B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-1B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-1B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 1 --model-format awq --quantization ${quantization}


Model Spec 3 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-2B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-2B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-2B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 4 (awq, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 2
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-2B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-2B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-2B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 2 --model-format awq --quantization ${quantization}


Model Spec 5 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-8B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 6 (awq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-8B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-8B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-8B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 8 --model-format awq --quantization ${quantization}


Model Spec 7 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-9B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-9B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-9B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 8 (awq, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 9
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-9B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-9B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-9B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 9 --model-format awq --quantization ${quantization}


Model Spec 9 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-14B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-14B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-14B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 14 --model-format pytorch --quantization ${quantization}


Model Spec 10 (awq, 14 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 14
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-14B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-14B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-14B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 14 --model-format awq --quantization ${quantization}


Model Spec 11 (pytorch, 38 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 38
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-38B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-38B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-38B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 38 --model-format pytorch --quantization ${quantization}


Model Spec 12 (awq, 38 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 38
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-38B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-38B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-38B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 38 --model-format awq --quantization ${quantization}


Model Spec 13 (pytorch, 78 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 78
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-78B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-78B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-78B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 78 --model-format pytorch --quantization ${quantization}


Model Spec 14 (awq, 78 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 78
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL3-78B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL3-78B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL3-78B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL3 --size-in-billions 78 --model-format awq --quantization ${quantization}

