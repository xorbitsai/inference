.. _models_builtin_whisper-base:

============
whisper-base
============

- **Model Name:** whisper-base
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-base
- **MLX model ID:** mlx-community/whisper-base-mlx

Execute the following command to launch the model::

   xinference launch --model-name whisper-base --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``