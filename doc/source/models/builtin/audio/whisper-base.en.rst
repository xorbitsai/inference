.. _models_builtin_whisper-base.en:

===============
whisper-base.en
===============

- **Model Name:** whisper-base.en
- **Model Family:** whisper
- **Abilities:** ['audio2text']
- **Multilingual:** False

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** openai/whisper-base.en
- **MLX model ID:** mlx-community/whisper-base.en-mlx

Execute the following command to launch the model::

   xinference launch --model-name whisper-base.en --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``