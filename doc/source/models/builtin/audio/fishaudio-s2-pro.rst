.. _models_builtin_fishaudio-s2-pro:

================
FishAudio-S2-Pro
================

- **Model Name:** FishAudio-S2-Pro
- **Model Family:** FishAudio
- **Abilities:** ['text2audio', 'text2audio_zero_shot', 'text2audio_voice_cloning']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **PyTorch model ID:** fishaudio/s2-pro
- **MLX model ID:** mlx-community/fish-audio-s2-pro-8bit

Execute the following command to launch the model::

   xinference launch --model-name FishAudio-S2-Pro --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``