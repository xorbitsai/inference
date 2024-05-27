.. _models_llm_seallm_v2.5:

========================================
seallm_v2.5
========================================

- **Context Length:** 8192
- **Model Name:** seallm_v2.5
- **Languages:** en, zh, vi, id, th, ms, km, lo, my, tl
- **Abilities:** generate
- **Description:** We introduce SeaLLM-7B-v2.5, the state-of-the-art multilingual LLM for Southeast Asian (SEA) languages

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: 
- **Model ID:** SeaLLMs/SeaLLM-7B-v2.5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/SeaLLMs/SeaLLM-7B-v2.5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seallm_v2.5 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (ggufv2, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 7
- **Quantizations:** Q4_K_M, Q8_0
- **Engines**: 
- **Model ID:** SeaLLMs/SeaLLM-7B-v2.5-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/SeaLLMs/SeaLLM-7B-v2.5-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name seallm_v2.5 --size-in-billions 7 --model-format ggufv2 --quantization ${quantization}

