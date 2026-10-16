.. _models_builtin_kokoro-82m:

==========
Kokoro-82M
==========

- **Model Name:** Kokoro-82M
- **Model Family:** Kokoro
- **Abilities:** ['text2audio', 'text2audio_zero_shot']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **PyTorch model ID:** hexgrad/Kokoro-82M
- **MLX model ID:** prince-canuma/Kokoro-82M

Execute the following command to launch the model::

   xinference launch --model-name Kokoro-82M --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``