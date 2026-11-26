.. _models_builtin_sensenova-u1.6-8b-mot-native:

============================
SenseNova-U1.6-8B-MoT-Native
============================

- **Model Name:** SenseNova-U1.6-8B-MoT-Native
- **Model Family:** sensenova_u1
- **Abilities:** text2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** sensenova/SenseNova-U1.6-8B-MoT-Native

Execute the following command to launch the model::

   xinference launch --model-name SenseNova-U1.6-8B-MoT-Native --model-type image


Select a checkpoint with ``--quantization``. Available values:
BF16, FP8-E4M3FN-W8A8, INT8-ConvRot-W8A8. The default is
``BF16``::

   xinference launch --model-name SenseNova-U1.6-8B-MoT-Native --model-type image --quantization INT8-ConvRot-W8A8

FP8 and INT8 require CUDA, BF16 compute, and ``vram_mode=full``.
