.. _models_llm_llama-3.2-vision:

================
llama-3.2-vision
================

- **Context Length:** 131072
- **Model Name:** llama-3.2-vision
- **Languages:** en, de, fr, it, pt, hi, es, th
- **Abilities:** generate, vision
- **Description:** The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The models outperform many of the available open source and closed multimodal models on common industry benchmarks...

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 11 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 11
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** meta-llama/Meta-Llama-3.2-11B-Vision
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.2-11B-Vision>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.2-11B-Vision>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine transformers --model-name llama-3.2-vision --size-in-billions 11 --model-format pytorch --quantization ${quantization}
   xinference launch --model-engine vllm --enforce_eager --max_num_seqs 16 --model-name llama-3.2-vision --size-in-billions 11 --model-format pytorch

Model Spec 2 (pytorch, 90 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 90
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** meta-llama/Meta-Llama-3.2-90B-Vision
- **Model Hubs**:  `Hugging Face <https://huggingface.co/meta-llama/Meta-Llama-3.2-90B-Vision>`__, `ModelScope <https://modelscope.cn/models/LLM-Research/Meta-Llama-3.2-90B-Vision>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine transformers --model-name llama-3.2-vision --size-in-billions 90 --model-format pytorch --quantization ${quantization}
   xinference launch --model-engine vllm --enforce_eager --max_num_seqs 16 --model-name llama-3.2-vision --size-in-billions 90 --model-format pytorch

