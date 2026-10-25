.. _models_builtin_sensevoicesmall:

===============
SenseVoiceSmall
===============

- **Model Name:** SenseVoiceSmall
- **Model Family:** funasr
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **PyTorch model ID:** FunAudioLLM/SenseVoiceSmall
- **MLX model ID:** mlx-community/SenseVoiceSmall

Execute the following command to launch the model::

   xinference launch --model-name SenseVoiceSmall --model-type audio --model-engine PyTorch

Available engines
^^^^^^^^^^^^^^^^^

* ``PyTorch``
* ``MLX``