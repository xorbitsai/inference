.. _models_llm_platypus2-70b-instruct:

========================================
platypus2-70b-instruct
========================================

- **Context Length:** 4096
- **Model Name:** platypus2-70b-instruct
- **Languages:** en
- **Abilities:** generate
- **Description:** Platypus-70B-instruct is a merge of garage-bAInd/Platypus2-70B and upstage/Llama-2-70b-instruct-v2.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 70 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 70
- **Quantizations:** none
- **Engines**: 
- **Model ID:** garage-bAInd/Platypus2-70B-instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/garage-bAInd/Platypus2-70B-instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name platypus2-70b-instruct --size-in-billions 70 --model-format pytorch --quantization ${quantization}

