.. _models_builtin_astra:

=====
Astra
=====

- **Model Name:** Astra
- **Model Family:** Astra
- **Abilities:** image2world
- **Engine:** PyTorch

Specifications
^^^^^^^^^^^^^^

- **Hugging Face Model ID:** EvanEternal/Astra
- **ModelScope Model ID:** Xorbits/Astra
- **Source:** https://github.com/EternalEvan/Astra
- **Officially documented hardware:** one 24 GB GPU, such as an RTX 3090

Launch the model with::

   xinference launch --model-name Astra --model-type world --model-engine PyTorch

Camera trajectories are selected with ``cam_type`` values from 1 through 7 and
are passed through ``extra_body`` in the REST API or ``model_kwargs`` in the
Python client.
