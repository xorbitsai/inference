.. _models_llm_deepseek-r1-0528-qwen3:

========================================
deepseek-r1-0528-qwen3
========================================

- **Context Length:** 131072
- **Model Name:** deepseek-r1-0528-qwen3
- **Languages:** en, zh
- **Abilities:** chat, reasoning
- **Description:** The DeepSeek R1 model has undergone a minor version upgrade, with the current version being DeepSeek-R1-0528. In the latest update, DeepSeek R1 has significantly improved its depth of reasoning and inference capabilities by leveraging increased computational resources and introducing algorithmic optimization mechanisms during post-training. The model has demonstrated outstanding performance across various benchmark evaluations, including mathematics, programming, and general logic. Its overall performance is now approaching that of leading models, such as O3 and Gemini 2.5 Pro

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 8
- **Quantizations:** none
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-0528-qwen3 --size-in-billions 8 --model-format pytorch --quantization ${quantization}


Model Spec 2 (gptq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 8
- **Quantizations:** Int4-W4A16, Int8-W8A16
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** QuantTrio/DeepSeek-R1-0528-Qwen3-8B-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/DeepSeek-R1-0528-Qwen3-8B-{quantization}>`__, `ModelScope <https://modelscope.cn/models/tclf90/DeepSeek-R1-0528-Qwen3-8B-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-0528-qwen3 --size-in-billions 8 --model-format gptq --quantization ${quantization}


Model Spec 3 (gptq, 8 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 8
- **Quantizations:** Int4-Int8Mix
- **Engines**: vLLM, Transformers, SGLang
- **Model ID:** QuantTrio/DeepSeek-R1-0528-Qwen3-8B-GPTQ-Int4-Int8Mix
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/DeepSeek-R1-0528-Qwen3-8B-GPTQ-Int4-Int8Mix>`__, `ModelScope <https://modelscope.cn/models/tclf90/DeepSeek-R1-0528-Qwen3-8B-GPTQ-Int4-Int8Mix>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-r1-0528-qwen3 --size-in-billions 8 --model-format gptq --quantization ${quantization}

