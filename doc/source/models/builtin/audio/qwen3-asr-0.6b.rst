.. _models_builtin_qwen3-asr-0.6b:

==============
Qwen3-ASR-0.6B
==============

- **Model Name:** Qwen3-ASR-0.6B
- **Model Family:** qwen3_asr
- **Abilities:** ['audio2text']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** Qwen/Qwen3-ASR-0.6B
- **MLX model ID:** mlx-community/Qwen3-ASR-0.6B-8bit

Execute the following command to launch the model::

   xinference launch --model-name Qwen3-ASR-0.6B --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``MLX``