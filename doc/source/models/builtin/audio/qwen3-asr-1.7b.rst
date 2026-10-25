.. _models_builtin_qwen3-asr-1.7b:

==============
Qwen3-ASR-1.7B
==============

- **Model Name:** Qwen3-ASR-1.7B
- **Model Family:** qwen3_asr
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** Qwen/Qwen3-ASR-1.7B
- **MLX model ID:** mlx-community/Qwen3-ASR-1.7B-8bit

Execute the following command to launch the model::

   xinference launch --model-name Qwen3-ASR-1.7B --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``