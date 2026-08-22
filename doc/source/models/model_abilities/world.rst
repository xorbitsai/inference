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
       "image": "https://example.com/start.png",
       "generation_config": {"response_format": "url"},
       "extra_body": {"num_frames": 97}
     }'

Generate with the Xinference Python client:

.. code-block:: python

   from xinference.client import Client

   client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
   model = client.get_model("<MODEL_UID>")
   result = model.generate(
       prompt="move forward through the scene",
       image="start.png",
       generation_config={"response_format": "url"},
       num_frames=97,
   )

The response follows the existing video result shape and contains either a
``url`` or ``b64_json`` value. ``Matrix-Game-3.0-5B`` and ``Astra`` require an
image; ``HY-WorldPlay-5B`` accepts text-only or image-conditioned generation.
None of the initial adapters accepts the reserved ``video`` input yet.

For Astra, camera control remains model-specific and is passed through
``extra_body`` or ``model_kwargs``. For example, ``cam_type`` accepts values
from 1 through 7 for the upstream preset trajectories. Astra's official
inference path runs on a single 24 GB NVIDIA GPU.
