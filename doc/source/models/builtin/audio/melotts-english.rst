.. _models_builtin_melotts-english:

===============
MeloTTS-English
===============

- **Model Name:** MeloTTS-English
- **Model Family:** MeloTTS
- **Abilities:** ['text2audio', 'text2audio_zero_shot']
- **Multilingual:** False

Specifications
^^^^^^^^^^^^^^

- **PyTorch model ID:** myshell-ai/MeloTTS-English
- **MLX model ID:** mlx-community/MeloTTS-English-MLX

Execute the following command to launch the model::

   xinference launch --model-name MeloTTS-English --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``