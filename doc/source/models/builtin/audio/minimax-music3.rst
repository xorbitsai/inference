.. _models_builtin_minimax-music3:

==============
MiniMax-Music3
==============

- **Model Name:** MiniMax-Music3
- **Model Family:** minimax_music3
- **Abilities:** ['text2music']
- **Multilingual:** True
- **Accelerator:** NVIDIA CUDA only
- **License:** `MiniMax-Music3 Community License <https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE>`_

Specifications
^^^^^^^^^^^^^^

- **Model ID:** MiniMaxAI/MiniMax-Music3
- **Engine:** diffusers

MiniMax-Music3 generates songs from lyrics plus a music description. Xinference
uses the Diffusers ``ModularPipeline`` backend and does not install or proxy
SGLang-Omni. The per-model virtual environment pins the Diffusers commit that
first included the pipeline (`2da7040be1a2e5f2fcbc8b985083342a308f5a86
<https://github.com/huggingface/diffusers/commit/2da7040be1a2e5f2fcbc8b985083342a308f5a86>`_)
because Diffusers 0.39.0 predates that integration.

Launch the model on an NVIDIA CUDA worker::

   xinference launch --model-name MiniMax-Music3 --model-type audio --model-engine diffusers

The default load configuration uses BF16 and keeps the pipeline on CUDA. Group
offload can be enabled for lower VRAM usage (at the cost of speed)::

   xinference launch --model-name MiniMax-Music3 --model-type audio \
     --model-engine diffusers --group_offload true

See :ref:`MiniMax-Music3 speech usage <minimax_music3_speech>` for request
examples and parameter limits. The model weights are distributed under the
MiniMax-Music3 Community License, not Apache-2.0.
