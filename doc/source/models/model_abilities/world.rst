.. _world:

====================
World (Experimental)
====================

World models generate an environment video from a prompt and, when supported,
an initial image. Xinference exposes one synchronous API for the first version:

``POST /v1/worlds/generations``

The common request fields are ``model``, ``prompt``, optional ``image`` or
``video``, and ``generation_config``. Model-specific controls such as action or
pose are passed through ``extra_body``. Continuous sessions and dedicated
action or streaming APIs are not part of this first version.

Supported models
----------------

* :ref:`Matrix-Game-3.0-5B <models_builtin_matrix-game-3.0-5b>`
* :ref:`HY-WorldPlay-5B <models_builtin_hy-worldplay-5b>`
* :ref:`Astra <models_builtin_astra>`

World models use the standard Xinference engine selection flow. Query
``GET /v1/engines/world/<MODEL_NAME>`` to list available engines and choose one
with ``model_engine`` when launching the model. The initial built-in engine is
``PyTorch`` and requires an NVIDIA CUDA GPU.

Quickstart
----------

Launch a model::

   xinference launch \
     --model-name Matrix-Game-3.0-5B \
     --model-type world \
     --model-engine PyTorch

Generate with cURL:

.. code-block:: bash

   curl -X POST \
     http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/worlds/generations \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "<MODEL_UID>",
       "prompt": "move forward through the scene",
       "image": "data:image/png;base64,<BASE64_DATA>",
       "generation_config": {"response_format": "b64_json"},
       "extra_body": {
         "num_frames": 97,
         "request_id": "world-request-1"
       }
     }'

Generate with the Xinference Python client:

.. code-block:: python

   from xinference.client import Client

   client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
   model = client.get_model("<MODEL_UID>")
   result = model.generate(
       prompt="move forward through the scene",
       image="start.png",
       generation_config={"response_format": "b64_json"},
       num_frames=97,
       request_id="world-request-1",
   )

The response follows the existing video result shape. The generated item is
``result["data"][0]`` and contains either a ``url`` or ``b64_json`` value.
``b64_json`` is the raw base64 payload without a ``data:video/...`` prefix.
The public REST endpoint accepts uploaded media
as base64 data URLs; the Python client converts local file paths or bytes to
that representation. Remote HTTP URLs and server-local paths are intentionally
not accepted by the public endpoint. ``Matrix-Game-3.0-5B`` and ``Astra``
require an image; ``HY-WorldPlay-5B`` accepts text-only or image-conditioned
generation. None of the initial adapters accepts the reserved ``video`` input
yet.

Progress and cancellation
-------------------------

Long-running requests can include a caller-generated ``request_id`` in
``extra_body`` (or ``model_kwargs`` in the Python client). While that request
is running, poll its progress::

   client.get_progress("world-request-1")

or use the REST endpoint::

   GET /v1/requests/world-request-1/progress

The progress response is ``{"progress": <float>}``, where the value ranges
from 0 to 1. Completed progress records are retained for five minutes by
default (configurable with ``XINFERENCE_REMOVE_PROGRESS_INTERVAL``).

Abort the same request with::

   client.abort_request("<MODEL_UID>", "world-request-1")

or::

   POST /v1/models/<MODEL_UID>/requests/world-request-1/abort
   {"block_duration": 30}

For remote clients, ``b64_json`` is directly consumable but materializes the
complete result in the response. ``url`` currently returns a path on the
worker that ran the model and is intended for clients that share that
filesystem. Xinference does not currently copy or clean up these files.
Artifact storage, public download URLs, and continuous streaming are separate
future capabilities rather than part of this synchronous first version.

The first launch downloads the pinned adapter source from its GitHub revision
and any auxiliary base weights required by the selected model. Selecting
ModelScope changes the model-weight source; it does not mirror the adapter
source checkout. Launch therefore requires access to both configured sources.
A fully managed air-gapped staging lifecycle for this complete dependency set
is not part of the first version.

For Astra, camera control remains model-specific and is passed through
``extra_body`` or ``model_kwargs``. For example, ``cam_type`` accepts values
from 1 through 7 for the upstream preset trajectories. Astra's official
inference path runs on a single 24 GB NVIDIA GPU.
