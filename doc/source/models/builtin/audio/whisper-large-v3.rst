.. _models_builtin_whisper-large-v3:

================
whisper-large-v3
================

- **Model Name:** whisper-large-v3
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-large-v3
- **MLX model ID:** mlx-community/whisper-large-v3-mlx

Execute the following command to launch the model::

   xinference launch --model-name whisper-large-v3 --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``