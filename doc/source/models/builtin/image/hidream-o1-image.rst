.. _models_builtin_hidream-o1-image:

================
HiDream-O1-Image
================

- **Model Name:** HiDream-O1-Image
- **Model Family:** hidream_o1
- **Abilities:** text2image, image2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** HiDream-ai/HiDream-O1-Image
- **Sources:** Hugging Face (``main``), ModelScope (``master``)
- **Inference steps:** 50
- **Required hardware:** NVIDIA CUDA GPU
- **PyTorch:** 2.10 or newer

Launch the model with::

   xinference launch --model-name HiDream-O1-Image --model-type image

The model supports text-to-image generation, instruction-based image editing,
and multiple reference images. It follows the official HiDream-O1 inference
recipe and requires PyTorch 2.10 or newer. Flash Attention is optional; the
default Xinference configuration uses the compatible non-Flash-Attention path.
