.. _models_builtin_whisper-tiny.en:

===============
whisper-tiny.en
===============

- **Model Name:** whisper-tiny.en
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** False

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-tiny.en
- **MLX model ID:** mlx-community/whisper-tiny.en-mlx

Execute the following command to launch the model::

   xinference launch --model-name whisper-tiny.en --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``