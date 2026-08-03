.. _models_download:

================
Download Sources
================

Xinference supports downloading various models from different sources.

Automatic Detection
^^^^^^^^^^^^^^^^^^^

By default, Xinference automatically decides between Hugging Face and ModelScope when
launching a model: it probes whether the Hugging Face endpoint is reachable (mirrors set
via ``HF_ENDPOINT`` and proxies set via ``HTTP_PROXY`` / ``HTTPS_PROXY`` are honored).
If it is reachable, models are downloaded from Hugging Face; otherwise Xinference falls
back to ModelScope. An HTTP error response (for example ``403``/``407`` from a blocking
corporate proxy, or a ``5xx`` from a broken mirror) counts as unreachable, since
downloads would fail anyway. The detection result is cached, so the probe runs at most once per
process, and its timeout can be tuned via the ``XINFERENCE_HUB_DETECT_TIMEOUT``
environment variable (default: 3 seconds).

You can also request the detection explicitly by passing ``--download_hub auto`` when
launching a model, or setting ``XINFERENCE_MODEL_SRC=auto``.

To pin a download source instead of relying on detection, set ``XINFERENCE_MODEL_SRC``
to ``huggingface`` or ``modelscope``, or pass ``--download_hub`` when launching a model.

HuggingFace
^^^^^^^^^^^^^^
Xinference downloads the required models from the official `Hugging Face model repository <https://huggingface.co/models>`_ when it is reachable.

.. note::
   If you have trouble connecting to Huggingface, you can use a mirror website to download with setting the environment variable ``HF_ENDPOINT=https://hf-mirror.com``.


ModelScope
^^^^^^^^^^^^^^

When the Hugging Face endpoint is not reachable (for example, no proxy is available),
Xinference automatically falls back to downloading models from
`ModelScope <https://modelscope.cn/models>`_.

You can also force this by manually setting an environment variable ``XINFERENCE_MODEL_SRC=modelscope``.

Please check the detail page of a model to confirm whether the model supports downloading from ModelScope.
If a model spec supports downloading from ModelScope, the "Model Hubs" section in the spec information will
include "ModelScope".
