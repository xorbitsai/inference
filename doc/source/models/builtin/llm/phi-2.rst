.. _models_llm_phi-2:

========================================
phi-2
========================================

- **Context Length:** 2048
- **Model Name:** phi-2
- **Languages:** en
- **Abilities:** generate
- **Description:** Phi-2 is a 2.7B Transformer based LLM used for research on model safety, trained with data similar to Phi-1.5 but augmented with synthetic texts and curated websites.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (ggufv2, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 2
- **Quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** TheBloke/phi-2-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/TheBloke/phi-2-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name phi-2 --size-in-billions 2 --model-format ggufv2 --quantization ${quantization}


Model Spec 2 (pytorch, 2 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** microsoft/phi-2
- **Model Hubs**:  `Hugging Face <https://huggingface.co/microsoft/phi-2>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name phi-2 --size-in-billions 2 --model-format pytorch --quantization ${quantization}

