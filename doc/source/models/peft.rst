.. _models_peft:

============
PEFT Adaptor
============

Parameter-Efficient Fine Tuning (PEFT) techniques involve the practice of retaining the fixed parameters of a pre-trained model while fine-tuning 
and introducing a limited count of adaptable parameters (referred to as adapters) to augment it.

Typical PEFT method includes Low-Rank Adaptation (LoRA), where the weights of the pre-trained model remain fixed, and further trainable components in the form of rank-decomposition matrices 
are introduced into each transformer block. 

Xinference provides a flexible and comprehensive way to integrate PEFT adaptors. You can integrate the PEFT adaptors on Hugging Face to your model to enhance its ability.

Launch a Model with a PEFT Model / Adaptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   xinference launch --model-name "llama-2-chat" --size-in-billions 7 --model-format pytorch --quantization none --peft-model-id "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA"

You can specify the Hugging Face ID for the adaptor as :code:`peft_model_id`. Make sure that the model format is :code:`pytorch`, and other model attributes (e.g., model size) are matched for the base model and the PEFT adaptor.

In addition, We cannot merge LORA layers when the model is loaded in 8-bit mode. Thus, make sure that the model quantization is not 8-bit.

For those PEFT adaptors extracted from Hugging Face, the adaptor will be automatically saved in :code:`${USER}/.xinference/peft_model/`. 

Example Usage
~~~~~~~~~~~~~

Now let us try integerating Llama2-Chinese-7b-Chat-LoRA on the Llama 2 7B Chat base model. Llama2-Chinese-7b-Chat-LoRA use a Chinese instruction set to fine-tune the Llama 2 Chat model, thus enhancing its proficiency in engaging in meaningful conversations in Chinese.

Before applying the LoRA:





After applying the LoRA:








