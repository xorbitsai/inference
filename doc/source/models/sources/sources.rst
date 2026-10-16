.. _models_download:

================
Download Sources
================

Xinference supports downloading various models from different sources.

Automatic Detection
^^^^^^^^^^^^^^^^^^^

By default, Xinference automatically decides between Hugging Face and ModelScope when
launching a model: it probes whether the Hugging Face endpoint is reachable (mirrors set
via ``HF_ENDPOINT`` are honored). If requests to that endpoint would use a proxy
configured for its URL scheme, automatic detection selects ModelScope without probing so
that large model downloads do not consume proxy traffic. For the default HTTPS endpoint,
this means ``HTTPS_PROXY`` or ``ALL_PROXY``; ``HTTP_PROXY`` applies to an HTTP
``HF_ENDPOINT``. ``NO_PROXY`` is honored, so an endpoint excluded from the proxy is still
probed directly and selects Hugging Face when reachable. An HTTP error response counts as
unreachable, since downloads would fail anyway. When Hugging Face offline mode is enabled
(``HF_HUB_OFFLINE=1`` or ``TRANSFORMERS_OFFLINE=1``), no probe runs and Hugging Face is
selected directly, so air-gapped deployments keep reading from their pre-populated local
Hugging Face cache. The detection result is cached, so the probe runs at most once per
process, and its timeout can be tuned via the ``XINFERENCE_HUB_DETECT_TIMEOUT``
environment variable (default: 3 seconds).

You can also request the detection explicitly by passing ``--download_hub auto`` when
launching a model, or setting ``XINFERENCE_MODEL_SRC=auto``.

To pin a download source instead of relying on detection, set ``XINFERENCE_MODEL_SRC``
to ``huggingface`` or ``modelscope``, or pass ``--download_hub`` when launching a model.
An explicit ``huggingface`` selection bypasses automatic proxy avoidance; the per-launch
option takes precedence over the service-level environment variable.

HuggingFace
^^^^^^^^^^^^^^
Xinference downloads the required models from the official `Hugging Face model repository <https://huggingface.co/models>`_ when it is reachable.

.. note::
   If you have trouble connecting to Huggingface, you can use a mirror website to download with setting the environment variable ``HF_ENDPOINT=https://hf-mirror.com``.


ModelScope
^^^^^^^^^^^^^^

When the Hugging Face endpoint would use an environment-configured proxy or is not
directly reachable, Xinference automatically falls back to downloading models from
`ModelScope <https://modelscope.cn/models>`_.

You can also force this by manually setting an environment variable ``XINFERENCE_MODEL_SRC=modelscope``.

Please check the detail page of a model to confirm whether the model supports downloading from ModelScope.
If a model spec supports downloading from ModelScope, the "Model Hubs" section in the spec information will
include "ModelScope".
