.. _models_llm_internvl2.5:

========================================
InternVL2.5
========================================

- **Context Length:** 16384
- **Model Name:** InternVL2.5
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** InternVL 2.5 is an open-source multimodal large language model (MLLM) to bridge the capability gap between open-source and proprietary commercial models in multimodal understanding. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2_5-1B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-1B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-1B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2_5-2B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-2B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-2B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2_5-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-4B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 4 (awq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 4
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL2_5-4B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-4B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-4B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 4 --model-format awq --quantization ${quantization}


Model Spec 5 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2_5-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-8B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 6 (awq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 8
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL2_5-8B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-8B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-8B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 8 --model-format awq --quantization ${quantization}


Model Spec 7 (pytorch, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 26
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2_5-26B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-26B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-26B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 26 --model-format pytorch --quantization ${quantization}


Model Spec 8 (awq, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 26
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL2_5-26B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-26B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-26B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 26 --model-format awq --quantization ${quantization}


Model Spec 9 (pytorch, 38 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 38
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2_5-38B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-38B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-38B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 38 --model-format pytorch --quantization ${quantization}


Model Spec 10 (awq, 38 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 38
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL2_5-38B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-38B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-38B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 38 --model-format awq --quantization ${quantization}


Model Spec 11 (pytorch, 78 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 78
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2_5-78B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-78B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-78B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 78 --model-format pytorch --quantization ${quantization}


Model Spec 12 (awq, 78 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 78
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** OpenGVLab/InternVL2_5-78B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2_5-78B-AWQ>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2_5-78B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name InternVL2.5 --size-in-billions 78 --model-format awq --quantization ${quantization}

