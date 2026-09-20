.. _models_builtin_breeze-tts-2:

============
Breeze-TTS-2
============

- **Model Name:** Breeze-TTS-2
- **Model Family:** Breeze-TTS-2
- **Abilities:** ['text2audio', 'text2audio_voice_design', 'text2audio_voice_cloning']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **PyTorch model ID:** BreezeBlue/Breeze-TTS-2
- **MLX model ID:** mlx-community/Breeze-TTS-2-mlx-8bit

Execute the following command to launch the model::

   xinference launch --model-name Breeze-TTS-2 --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``