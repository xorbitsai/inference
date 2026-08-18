.. _models_builtin_ideogram4:

==========
Ideogram4
==========

- **Model Name:** Ideogram4
- **Model Family:** stable_diffusion
- **Abilities:** text2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** ideogram-ai/ideogram-4-nf4-diffusers
- **Quantization:** NF4
- **Hardware:** NVIDIA CUDA GPU

The checkpoint uses the Ideogram 4 Non-Commercial Model Agreement and its
repository is gated. Accept the license on the selected model hub and
authenticate with that hub before launching. Ideogram 4 accepts plain text
prompts, but serialized structured JSON captions provide the best quality and
control.

Execute the following command to launch the model::

   xinference launch --model-name Ideogram4 --model-type image

To download from ModelScope for an individual launch::

   xinference launch --model-name Ideogram4 --model-type image --download_hub modelscope
