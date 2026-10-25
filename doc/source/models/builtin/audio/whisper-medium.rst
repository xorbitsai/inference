.. _models_builtin_whisper-medium:

==============
whisper-medium
==============

- **Model Name:** whisper-medium
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-medium
- **MLX model ID:** mlx-community/whisper-medium-mlx

Execute the following command to launch the model::

   xinference launch --model-name whisper-medium --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``