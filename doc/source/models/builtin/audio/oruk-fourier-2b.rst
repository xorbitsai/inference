.. _models_builtin_oruk-fourier-2b:

===============
Oruk-Fourier-2B
===============

- **Model Name:** Oruk-Fourier-2B
- **Model Family:** qwen3_asr
- **Abilities:** ['audio2text']
- **Multilingual:** False

Specifications
^^^^^^^^^^^^^^

- **Model ID:** oruk/oruk-fourier-2b

.. note::

   Requires Linux with an NVIDIA CUDA GPU (uses the ``vLLM`` engine).

Execute the following command to launch the model::

   xinference launch --model-name Oruk-Fourier-2B --model-type audio --model-engine vLLM