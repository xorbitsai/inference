.. _models_builtin_minimax-h3:

==========
MiniMax-H3
==========

- **Model Name:** MiniMax-H3
- **Model Family:** MiniMax-H3
- **Abilities:** text2video, image2video, firstlastframe2video

Specifications
^^^^^^^^^^^^^^

- **Model ID:** MiniMaxAI/MiniMax-H3
- **Lightning Model ID:** lightx2v/Minimax-h3-Turbo
- **Lightning Versions:** 4step_v0.1, 8step_v1.0_bf16, 4step_v1.0_768p_bf16

Execute the following command to launch the model::

   xinference launch --model-name MiniMax-H3 --model-type video

For Lightning LoRA acceleration, use::

   xinference launch --model-name MiniMax-H3 --model-type video --lightning_version ${lightning_version}