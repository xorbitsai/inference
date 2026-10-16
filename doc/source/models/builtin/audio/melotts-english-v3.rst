.. _models_builtin_melotts-english-v3:

==================
MeloTTS-English-v3
==================

- **Model Name:** MeloTTS-English-v3
- **Model Family:** MeloTTS
- **Abilities:** ['text2audio', 'text2audio_zero_shot']
- **Multilingual:** False

Specifications
^^^^^^^^^^^^^^

- **PyTorch model ID:** myshell-ai/MeloTTS-English-v3
- **MLX model ID:** mlx-community/MeloTTS-English-v3-MLX

Execute the following command to launch the model::

   xinference launch --model-name MeloTTS-English-v3 --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``