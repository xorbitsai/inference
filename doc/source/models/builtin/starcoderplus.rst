.. _models_builtin_starcoderplus:

=============
StarCoderPlus
=============

- **Model Name:** starcoderplus
- **Languages:** en
- **Abilities:** embed, generate

Specifications
^^^^^^^^^^^^^^

Model Spec (pytorch, 16 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 16
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** bigcode/starcoderplus

Execute the following command to launch the model, remember to replace `${quantization}` with your
chosen quantization method from the options listed above::

   xinference launch --model-name starcoderplus --size-in-billions 16 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit quantization is not supported on macOS.
