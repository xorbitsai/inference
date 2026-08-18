.. _models_builtin_hidream-o1-image-dev:

====================
HiDream-O1-Image-Dev
====================

- **Model Name:** HiDream-O1-Image-Dev
- **Model Family:** hidream_o1
- **Abilities:** text2image, image2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** HiDream-ai/HiDream-O1-Image-Dev
- **Sources:** Hugging Face (``main``), ModelScope (``master``)
- **Inference steps:** 28
- **Required hardware:** NVIDIA CUDA GPU
- **PyTorch:** 2.10 or newer

Launch the model with::

   xinference launch --model-name HiDream-O1-Image-Dev --model-type image

This distilled variant uses the official 28-step schedule. Image editing uses
the recommended flow-matching scheduler by default; text-to-image generation
and multi-reference generation use the flash scheduler.
