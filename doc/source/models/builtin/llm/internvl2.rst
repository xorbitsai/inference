.. _models_llm_internvl2:

========================================
internvl2
========================================

- **Context Length:** 32768
- **Model Name:** internvl2
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** InternVL 2 is an open-source multimodal large language model (MLLM) to bridge the capability gap between open-source and proprietary commercial models in multimodal understanding. 

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 1 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2-1B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-1B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2-1B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 1 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2-2B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-2B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2-2B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 3 (awq, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 2
- **Quantizations:** Int4
- **Engines**: 
- **Model ID:** OpenGVLab/InternVL2-2B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-2B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 2 --model-format awq --quantization ${quantization}


Model Spec 4 (pytorch, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 4
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2-4B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-4B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2-4B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 4 --model-format pytorch --quantization ${quantization}


Model Spec 5 (awq, 4 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 4
- **Quantizations:** Int4
- **Engines**: 
- **Model ID:** OpenGVLab/InternVL2-8B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-8B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 4 --model-format awq --quantization ${quantization}


Model Spec 6 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-8B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2-8B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 7 (pytorch, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 26
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2-26B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-26B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2-26B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 26 --model-format pytorch --quantization ${quantization}


Model Spec 8 (awq, 26 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 26
- **Quantizations:** Int4
- **Engines**: 
- **Model ID:** OpenGVLab/InternVL2-26B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-26B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 26 --model-format awq --quantization ${quantization}


Model Spec 9 (pytorch, 40 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 40
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2-40B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-40B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2-40B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 40 --model-format pytorch --quantization ${quantization}


Model Spec 10 (awq, 40 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 40
- **Quantizations:** Int4
- **Engines**: 
- **Model ID:** OpenGVLab/InternVL2-40B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-40B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 40 --model-format awq --quantization ${quantization}


Model Spec 11 (pytorch, 76 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 76
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** OpenGVLab/InternVL2-Llama3-76B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B>`__, `ModelScope <https://modelscope.cn/models/OpenGVLab/InternVL2-Llama3-76B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 76 --model-format pytorch --quantization ${quantization}


Model Spec 12 (awq, 76 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 76
- **Quantizations:** Int4
- **Engines**: 
- **Model ID:** OpenGVLab/InternVL2-Llama3-76B-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name internvl2 --size-in-billions 76 --model-format awq --quantization ${quantization}

