.. _models_builtin_sensenova-u1.6-8b-mot-gguf:

==========================
SenseNova-U1.6-8B-MoT-GGUF
==========================

- **Model Name:** SenseNova-U1.6-8B-MoT-GGUF
- **Model Family:** sensenova_u1
- **Abilities:** text2image
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** sensenova/SenseNova-U1.6-8B-MoT-GGUF

Execute the following command to launch the model::

   xinference launch --model-name SenseNova-U1.6-8B-MoT-GGUF --model-type image


Select a checkpoint with ``--quantization``. Available values:
Q4_0, Q6_K, Q8_0. The default is
``Q4_0``::

   xinference launch --model-name SenseNova-U1.6-8B-MoT-GGUF --model-type image --quantization Q8_0
