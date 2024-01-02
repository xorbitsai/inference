.. _models_download:

================
Download Sources
================

Xinference supports downloading various models from different sources.

HuggingFace
^^^^^^^^^^^^^^
Xinference directly downloads the required models from the official `Hugging Face model repository <https://huggingface.co/models>`_ by default.

.. note::
   If you have trouble connecting to Huggingface, you can use a mirror website to download with setting the environment variable ``HF_ENDPOINT=https://hf-mirror.com``.


ModelScope
^^^^^^^^^^^^^^

When Xinference detects that the system's language is set to Simplified Chinese, it will automatically
set the model download source to `ModelScope <https://modelscope.cn/models>`_.

You can also achieve this by manually setting an environment variable ``XINFERENCE_MODEL_SRC=modelscope``.

Please check the detail page of a model to confirm whether the model supports downloading from ModelScope.
If a model spec supports downloading from ModelScope, the "Model Hubs" section in the spec information will
include "ModelScope".
