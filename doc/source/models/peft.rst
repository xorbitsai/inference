.. _models_peft:

=============
PEFT Adaptor
=============

Parameter-Efficient Fine Tuning (PEFT) techniques involve the practice of retaining the fixed parameters of a pre-trained model while fine-tuning 
and introducing a limited count of adaptable parameters (referred to as adapters) to augment it.

Typical PEFT method includes Low-Rank Adaptation (LoRA), where the weights of the pre-trained model remain fixed, and further trainable components in the form of rank-decomposition matrices 
are introduced into each transformer block. 

Xinference provides a flexible and comprehensive way to integrate PEFT adaptors. You can load the PEFT path trained by yourself onto the model, or you can use the PEFT path on Hugging Face.

Launch Model with PEFT Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   xinference launch --model-name "llama-2-chat" -s 7 -f "pytorch" -p "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA"

You can specify the hugging_face id for the adaptor as the PEFT path, or you can specify a PEFT path yourself. For the PEFT adaptors extracted from Hugging Face,
the adaptor will be saved in :code:`your_home_directory/.lora`