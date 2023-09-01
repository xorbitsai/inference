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

.. note::

   We cannot merge LORA layers when the model is loaded in 8-bit or 4-bit mode. Thus, make sure that the model quantization is none if you want to apply LoRA to it.

You can specify the Hugging Face ID for the adaptor as :code:`peft_model_id`. Make sure that the model format is :code:`pytorch`, and other model attributes (e.g., model size) are matched for the base model and the PEFT adaptor.

For those PEFT adaptors extracted from Hugging Face, the adaptor will be automatically saved in :code:`${USER}/.xinference/peft_model/`. 

Example Usage
~~~~~~~~~~~~~

Now let us try integerating Llama2-Chinese-7b-Chat-LoRA on the Llama 2 7B Chat base model. Llama2-Chinese-7b-Chat-LoRA use a Chinese instruction set to fine-tune the Llama 2 Chat model, thus enhancing its proficiency in engaging in meaningful conversations in Chinese.

Before applying the LoRA:

.. code-block:: bash

   $ xinference launch --model-name "llama-2-chat" --size-in-billions 7 --model-format pytorch --quantization none
   $ xinference chat --model-uid {model_uid_from_previous_command}

   User: 世界上最大的动物是什么？
   Assistant:  Great question! There are several contenders for the title of the largest animal in the world, depending on how you define "largest." Here are a few candidates:
   1. Blue whale: On average, blue whales can grow up to 82 feet (25 meters) in length and weigh around 150-170 tons (136,000-152,000 kg). They are the largest known animal to have ever lived on Earth, with some individuals reaching lengths of up to 100 feet (30 meters).
   2. Fin whale: Fin whales are the second-largest animal on Earth, with adults typically reaching lengths of around 85 feet (26 meters) and weighing up to 74 tons (67,000 kg).
   3. Hippopotamus: Despite their stocky build, hippopotamuses can be quite large, with males reaching lengths of up to 15 feet (4.5 meters) and weighing up to 4,000 pounds (1,800 kg).
   ...

The model replied to the question in English. 

After applying the LoRA, the model replied to the question in Chinese:

.. code-block:: bash

   $ xinference launch --model-name "llama-2-chat" --size-in-billions 7 --model-format pytorch --quantization none --peft-model-id "FlagAlpha/Llama2-Chinese-7b-Chat-LoRA"
   $ xinference chat --model-uid {model_uid_from_previous_command}










