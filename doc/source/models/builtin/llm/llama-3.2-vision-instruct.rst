.. _models_llm_llama-3.2-vision-instruct:

========================================
llama-3.2-vision-instruct
========================================

- **Context Length:** 131072
- **Model Name:** llama-3.2-vision-instruct
- **Languages:** en, de, fr, it, pt, hi, es, th
- **Abilities:** chat, vision
- **Description:** Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image...

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 11 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 11
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** meta-llama/Meta-Llama-3.2-11B-Vision-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.2-11B-Vision-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.2-11B-Vision-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.2-vision-instruct --size-in-billions 11 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 90 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 90
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** meta-llama/Meta-Llama-3.2-90B-Vision-Instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.2-90B-Vision-Instruct>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.2-90B-Vision-Instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name llama-3.2-vision-instruct --size-in-billions 90 --model-format pytorch --quantization ${quantization}

