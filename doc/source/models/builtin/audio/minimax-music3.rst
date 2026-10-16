.. _models_builtin_minimax-music3:

==============
MiniMax-Music3
==============

- **Model Name:** MiniMax-Music3
- **Model Family:** minimax_music3
- **Abilities:** ['text2music']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **diffusers (none) model ID:** MiniMaxAI/MiniMax-Music3
- **MLX (8-bit) model ID:** mlx-community/MiniMax-Music3-8bit

Execute the following command to launch the model::

   xinference launch --model-name MiniMax-Music3 --model-type audio --model-engine diffusers

Available engines
^^^^^^^^^^^^^^^^^

* ``diffusers``
* ``MLX``

Available quantizations by engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``diffusers``: ``none``
* ``MLX``: ``8-bit``