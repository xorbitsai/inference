.. _models_builtin_rwkv_4_pile:

=============
RWKV-4-Pile
=============

- **Model Name:** rwkv-4-pile
- **Languages:** en
- **Abilities:** generate
- **Description:** The RWKV (Receptance-Weighted Key-Value) models are a series of language models that ranges in size. They are noteworthy for their large-scale implementation and innovative architecture, which combines the strengths of RNN and Transformer models.

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 1 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 1
- **Quantizations:** none
- **Model ID:** RWKV/rwkv-4-430m-pile

Execute the following command to launch the model::

   xinference launch --model-name rwkv-4-pile --size-in-billions 1 --model-format pytorch --quantization none

Model Spec 2 (pytorch, 2 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 2
- **Quantizations:** none
- **Model ID:** RWKV/rwkv-4-1b5-pile

Execute the following command to launch the model::

   xinference launch --model-name rwkv-4-pile --size-in-billions 2 --model-format pytorch --quantization none

Model Spec 3 (pytorch, 3 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 3
- **Quantizations:** none
- **Model ID:** RWKV/rwkv-4-3b-pile

Execute the following command to launch the model::

   xinference launch --model-name rwkv-4-pile --size-in-billions 3 --model-format pytorch --quantization none

Model Spec 4 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Model ID:** RWKV/rwkv-4-7b-pile

Execute the following command to launch the model::

   xinference launch --model-name rwkv-4-pile --size-in-billions 7 --model-format pytorch --quantization none

Model Spec 5 (pytorch, 14 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Model ID:** RWKV/rwkv-4-14b-pile

Execute the following command to launch the model::

   xinference launch --model-name rwkv-4-pile --size-in-billions 14 --model-format pytorch --quantization none
