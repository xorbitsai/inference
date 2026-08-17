.. _models_builtin_whisper-large-v3-turbo:

======================
whisper-large-v3-turbo
======================

- **Model Name:** whisper-large-v3-turbo
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-large-v3-turbo
- **MLX model ID:** mlx-community/whisper-large-v3-turbo

Execute the following command to launch the model::

   xinference launch --model-name whisper-large-v3-turbo --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``