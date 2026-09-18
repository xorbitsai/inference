.. _models_builtin_irodori-tts-v4.1-small:

======================
Irodori-TTS-v4.1-Small
======================

- **Model Name:** Irodori-TTS-v4.1-Small
- **Model Family:** Irodori-TTS
- **Abilities:** ['text2audio', 'text2audio_voice_design', 'text2audio_voice_cloning', 'text2audio_emotion_control']
- **Multilingual:** False

Specifications
^^^^^^^^^^^^^^

- **PyTorch (none) model ID:** Aratako/Irodori-TTS-v4.1-Small
- **PyTorch (INT8-Weight-Only) model ID:** Aratako/Irodori-TTS-v4.1-Small-Quantized
- **PyTorch (INT8-Dynamic) model ID:** Aratako/Irodori-TTS-v4.1-Small-Quantized
- **PyTorch (INT4-Weight-Only) model ID:** Aratako/Irodori-TTS-v4.1-Small-Quantized
- **PyTorch (Float8-Weight-Only) model ID:** Aratako/Irodori-TTS-v4.1-Small-Quantized
- **PyTorch (Float8-Dynamic) model ID:** Aratako/Irodori-TTS-v4.1-Small-Quantized
- **MLX (none) model ID:** mlx-community/Irodori-TTS-v4.1-Small-8bit

Execute the following command to launch the model::

   xinference launch --model-name Irodori-TTS-v4.1-Small --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``

Available quantizations by engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``PyTorch``: ``none``, ``INT8-Weight-Only``, ``INT8-Dynamic``, ``INT4-Weight-Only``, ``Float8-Weight-Only``, ``Float8-Dynamic``
* ``MLX``: ``none``