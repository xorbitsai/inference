.. _models_builtin_whisper-small:

=============
whisper-small
=============

- **Model Name:** whisper-small
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-small
- **MLX model ID:** mlx-community/whisper-small-mlx

Execute the following command to launch the model::

   xinference launch --model-name whisper-small --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``