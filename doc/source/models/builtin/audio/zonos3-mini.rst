.. _models_builtin_zonos3-mini:

===========
ZONOS3-Mini
===========

- **Model Name:** ZONOS3-Mini
- **Model Family:** zonos3_mini
- **Abilities:** ['text2audio', 'text2audio_zero_shot', 'text2audio_voice_cloning']
- **Multilingual:** True

Specifications
^^^^^^^^^^^^^^

- **transformers model ID:** Zyphra/ZONOS3-Mini-hf
- **SGLang-Omni model ID:** Zyphra/ZONOS3-Mini
- **MLX model ID:** Zyphra/ZONOS3-Mini
- **GGML model ID:** Zyphra/ZONOS3-Mini

Execute the following command to launch the model::

   xinference launch --model-name ZONOS3-Mini --model-type audio --model-engine transformers

Available engines
^^^^^^^^^^^^^^^^^

* ``transformers``
* ``SGLang-Omni``
* ``MLX``
* ``GGML``
