.. _models_llm_qwen2-vl-instruct:

========================================
qwen2-vl-instruct
========================================

- **Context Length:** 32768
- **Model Name:** qwen2-vl-instruct
- **Languages:** en, zh
- **Abilities:** chat, vision
- **Description:** Qwen2-VL: To See the World More Clearly.Qwen2-VL is the latest version of the vision language models in the Qwen model familities.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-2B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 2 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 2
- **Quantizations:** Int8
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 2 --model-format gptq --quantization ${quantization}


Model Spec 3 (gptq, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 2
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 2 --model-format gptq --quantization ${quantization}


Model Spec 4 (awq, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 2
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-2B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-2B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 2 --model-format awq --quantization ${quantization}


Model Spec 5 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-7B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 6 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int8
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 7 (gptq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 7 --model-format gptq --quantization ${quantization}


Model Spec 8 (awq, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 7
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-7B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-7B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 7 --model-format awq --quantization ${quantization}


Model Spec 9 (pytorch, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 72
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-72B-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 72 --model-format pytorch --quantization ${quantization}


Model Spec 10 (awq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 72
- **Quantizations:** Int4
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-72B-Instruct-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-AWQ>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 72 --model-format awq --quantization ${quantization}


Model Spec 11 (gptq, 72 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 72
- **Quantizations:** Int4, Int8
- **Engines**: Transformers
- **Model ID:** Qwen/Qwen2-VL-72B-Instruct-GPTQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/qwen/Qwen2-VL-72B-Instruct-GPTQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name qwen2-vl-instruct --size-in-billions 72 --model-format gptq --quantization ${quantization}

