.. _models_builtin_whisper-small.en:

================
whisper-small.en
================

- **Model Name:** whisper-small.en
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** False

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-small.en
- **MLX model ID:** mlx-community/whisper-small.en-mlx

Execute the following command to launch the model::

   xinference launch --model-name whisper-small.en --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``