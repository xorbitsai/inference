.. _models_builtin_f5-tts:

======
F5-TTS
======

- **Model Name:** F5-TTS
- **Model Family:** F5-TTS
- **Abilities:** ['text2audio', 'text2audio_zero_shot', 'text2audio_voice_cloning']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **PyTorch model ID:** SWivid/F5-TTS
- **MLX model ID:** lucasnewman/f5-tts-mlx

Execute the following command to launch the model::

   xinference launch --model-name F5-TTS --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``