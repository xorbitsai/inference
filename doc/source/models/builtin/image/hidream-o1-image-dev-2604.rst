.. _models_builtin_hidream-o1-image-dev-2604:

=========================
HiDream-O1-Image-Dev-2604
=========================

- **Model Name:** HiDream-O1-Image-Dev-2604
- **Model Family:** hidream_o1
- **Abilities:** text2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** HiDream-ai/HiDream-O1-Image-Dev-2604
- **Sources:** Hugging Face (``main``), ModelScope (``master``)
- **Inference steps:** 28
- **Required hardware:** NVIDIA CUDA GPU
- **PyTorch:** 2.10 or newer

Launch the model with::

   xinference launch --model-name HiDream-O1-Image-Dev-2604 --model-type image

This checkpoint is specialized for text-to-image generation and uses the
official Dev-2604 float32 loading and sampling defaults.
