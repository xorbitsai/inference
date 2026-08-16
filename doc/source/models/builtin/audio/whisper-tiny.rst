.. _models_builtin_whisper-tiny:

============
whisper-tiny
============

- **Model Name:** whisper-tiny
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-tiny
- **MLX model ID:** mlx-community/whisper-tiny

Execute the following command to launch the model::

   xinference launch --model-name whisper-tiny --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``