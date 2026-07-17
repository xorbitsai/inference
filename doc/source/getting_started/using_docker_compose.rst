.. _using_docker_compose:

=========================
Docker Compose Deployment
=========================

Xinference ships an official Docker Compose setup for standalone deployment, located at
`xinference/deploy/docker <https://github.com/xorbitsai/inference/tree/main/xinference/deploy/docker>`_.
It supports online hosts as well as fully offline / air-gapped environments, where an optional
private PyPI server is started alongside Xinference to serve the packages installed at
model-launch time.

Prerequisites
=============
* Docker Compose **v2.24.4 or above** (required by the ``env_file: required: false`` and ``!reset`` features used in the compose files).
* For GPU deployment: a host with NVIDIA GPUs, CUDA installed, and `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_. See :ref:`using_docker_image` for image requirements.
* Get the whole ``xinference/deploy/docker`` directory. The compose file bind-mounts ``pip.conf`` from the same directory, so downloading ``docker-compose.yml`` alone is not sufficient:

.. code-block:: bash

   git clone https://github.com/xorbitsai/inference.git
   cd inference/xinference/deploy/docker


Quick Start (GPU)
=================
Start Xinference with all GPUs of the host:

.. code-block:: bash

   docker compose up -d

Wait until the service is healthy, then verify:

.. code-block:: bash

   docker compose ps
   curl http://localhost:9997/status

The Web UI is served at ``http://localhost:9997``.

.. note::

   Since v3.0, authentication is enabled by default: the first visit to the
   Web UI asks you to create the initial admin account, and API calls require
   a login or API key afterwards. To run without authentication, add
   ``XINFERENCE_AUTH_ADVANCED=false`` under the ``environment:`` section of
   ``docker-compose.yml``. See :ref:`user_guide_auth_system`.


Configuration
=============
All settings are exposed as variables with sensible defaults and can be overridden through a
``.env`` file next to ``docker-compose.yml``. Copy the template and edit as needed:

.. code-block:: bash

   cp .env.example .env

Available variables:

* ``XINFERENCE_IMAGE``: image to run, defaults to ``xprobe/xinference:latest``. Pin a release tag such as ``xprobe/xinference:v<version>`` for production.
* ``XINFERENCE_PORT``: host port of the RESTful API / Web UI, defaults to ``9997``.
* ``XINFERENCE_MODEL_SRC``: model download source, ``huggingface`` (default) or ``modelscope``.
* ``XINFERENCE_SHM_SIZE``: shared memory size, defaults to ``8gb``. Increase for multi-GPU inference.
* ``XINFERENCE_LOG_LEVEL``: log level, defaults to ``info``.
* ``XINFERENCE_HOME_DIR`` / ``XINFERENCE_HF_CACHE_DIR`` / ``XINFERENCE_MODELSCOPE_CACHE_DIR``: persistence locations. They default to named Docker volumes; point them at absolute host paths to reuse existing model caches, in the same way as described in :ref:`using_docker_image`.
* ``XINFERENCE_WHEELS_DIR`` / ``XINFERENCE_PYPISERVER_PORT`` / ``XINFERENCE_PYPISERVER_IMAGE``: offline profile settings, see below.

For other runtime options (authentication, OpenTelemetry, health-check tuning, ...), add the
corresponding variables under the ``environment:`` section of ``docker-compose.yml``.
See :ref:`environments` for the full list.


CPU-only Deployment
===================
On hosts without NVIDIA GPUs, apply the CPU override file, which switches the image to the
``-cpu`` variant and removes the GPU reservation:

.. code-block:: bash

   docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d


Health Check and Restart Policy
===============================
The ``xinference`` service reports health through the ``/status`` endpoint and restarts
automatically unless explicitly stopped (``restart: unless-stopped``). ``docker compose ps``
shows the health state; orchestration on top of compose can rely on it.


Offline / Air-gapped Deployment
===============================
By default Xinference installs the extra Python packages declared by a model at launch time
into a per-model virtual environment (controlled by ``XINFERENCE_ENABLE_VIRTUAL_ENV``,
see :ref:`environments`). On a host without Internet access these installs would fail.

The ``offline`` compose profile solves this by starting a private PyPI server next to
Xinference. Its image, ``xprobe/xinference-pypiserver``, ships the index-compatible wheels
the runtime may install into per-model virtual environments — including the ``vllm`` /
``sglang`` CUDA stacks — so no wheel preparation is needed for supported models. The offline
configuration points every runtime ``pip`` / ``uv`` invocation inside the Xinference container
at it.

.. note::

   The prebuilt mirror's GPU stack targets CUDA 13.0. This does not remove the runtime's
   existing online support for CUDA 12.8/12.9, but those stacks are not included in this
   mirror image.

Step 1: Transfer the Docker images
----------------------------------
Transfer the Docker images to the offline host (``docker save`` / ``docker load``): the
Xinference image and the mirror image. Pin both to the **same release tag** so the mirror
contents match that release's model specs and engine dependency lists, and record the pins
in ``.env``:

.. code-block:: bash

   XINFERENCE_IMAGE=xprobe/xinference:v2.9.0
   XINFERENCE_PYPISERVER_IMAGE=xprobe/xinference-pypiserver:v2.9.0

Step 2: Enable the offline configuration
----------------------------------------
.. code-block:: bash

   cp offline.env.example offline.env

Then open ``pip.conf`` and uncomment the three lines of the offline block:

.. code-block:: ini

   [global]
   index-url = http://xinference-pypiserver:8080/simple
   extra-index-url = http://xinference-pypiserver:8080/simple
   trusted-host = xinference-pypiserver

.. note::

   All three pieces are required because they cover different code paths. ``pip.conf`` feeds
   Xinference's pip-config inheritance, which passes the private index explicitly to the
   per-model virtual-env installer; the ``UV_*`` variables in ``offline.env`` cover ``uv``
   invocations that do not carry index flags (such as the dependency-resolution dry-run);
   ``XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL=1`` enables direct-wheel rewriting only for this
   self-contained mirror. A normal pip mirror configured by an online user does not enable
   that behavior.

Step 3: Start with the offline profile
--------------------------------------
.. code-block:: bash

   docker compose --profile offline up -d

Or set ``COMPOSE_PROFILES=offline`` in ``.env`` to omit the ``--profile`` flag. Combine with the
CPU override if needed:

.. code-block:: bash

   docker compose --profile offline -f docker-compose.yml -f docker-compose.cpu.yml up -d

The private index is also published on the host (default port ``8080``), so other machines on
the same network can reuse it with ``pip install -i http://<host>:8080/simple ...``.

.. note::

   When launching models with the **vLLM** or **SGLang** engines, Xinference by default resolves
   some dependencies from hardcoded public indexes (``wheels.vllm.ai``,
   ``download.pytorch.org``) and from direct wheel URLs (``sgl_kernel``). The offline
   ``pip.conf`` above overrides both: the private index replaces the public indexes, and
   direct wheel-URL requirements are resolved from it as ``name==version``. The baked mirror
   already carries these CUDA wheels. Alternatively, set ``XINFERENCE_ENABLE_VIRTUAL_ENV=0``
   in ``offline.env`` to skip runtime installs entirely and rely on the packages baked into
   the Xinference image.

   For the **llama.cpp** engine, the mirror carries the CPU build of ``xllamacpp`` from PyPI.
   Its GPU wheels live on a separate CUDA-specific index that is unavailable offline, so a GPU
   host falls back to the CPU build and logs a warning. To retain llama.cpp GPU acceleration,
   preinstall the matching ``xllamacpp`` GPU wheel in a custom runtime image.

.. warning::

   Model specifications containing ``git+`` or other non-wheel direct references cannot be
   represented faithfully by a Python simple index. In explicit offline-install mode,
   Xinference rejects them before attempting network egress and reports the offending
   requirement. The current built-ins in this category include the Transformers path of
   HunyuanOCR, MiniCPM-V-4.6, and MiniCPM-V-4.6-Thinking, plus FLUX.2-klein. Preinstall the
   required source revision in a custom image or replace it with an index-resolvable package
   before using these models air-gapped. The FlashInfer AOT repair fetched from its public
   wheel index is also skipped in explicit offline mode; Blackwell deployments that require it
   should bake those packages into a custom image.

Bring your own wheels (optional)
--------------------------------
To serve a self-curated wheel directory instead of the baked mirror — for example a small
subset for specific models — add the ``docker-compose.byo-wheels.yml`` override, which swaps
the image for the stock ``pypiserver/pypiserver:v2.3.2`` and mounts ``./wheels``:

Set ``XINFERENCE_BYO_PYPISERVER_IMAGE`` if that stock image has been mirrored into a private
registry. This setting is intentionally independent of ``XINFERENCE_PYPISERVER_IMAGE``, which
selects the prebuilt Xinference mirror in the normal offline profile.

.. code-block:: bash

   python3 -m pip download \
      --dest ./wheels \
      --only-binary=:all: \
      --python-version 312 \
      --platform manylinux2014_x86_64 \
      'transformers>=4.53.3' accelerate

   chmod -R a+rwX ./wheels   # pypiserver runs as UID 9898
   docker compose --profile offline \
      -f docker-compose.yml -f docker-compose.byo-wheels.yml up -d

Offline model weights
---------------------
The offline configuration sets ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``, so model
weights are read from the local cache only. Populate the cache before going offline, either by
pointing the cache variables in ``.env`` at host directories that already contain the models,
or by launching each model once on a connected host and copying the volumes. Models can also be
loaded from arbitrary local paths by registering custom models or passing ``--model-path``.

Enforced network isolation (optional)
-------------------------------------
The offline configuration redirects every download to local sources, but by itself it does not
prevent the containers from reaching the Internet if the host happens to have connectivity. To
*guarantee* isolation at the network level, add the air-gap override, which moves Xinference and
the private PyPI server onto an ``internal`` Docker network with no external routing:

.. code-block:: bash

   docker compose --profile offline \
      -f docker-compose.yml -f docker-compose.airgap.yml up -d

Because Docker does not publish ports of internal-only networks, the override adds a minimal
TCP gateway (``alpine/socat``) that forwards the API port from the host into the isolated
network. The gateway only relays inbound traffic to ``xinference:9997``; the containers cannot
use it as an egress path. Remember to transfer the ``alpine/socat`` image to the offline host
along with the others.

.. note::

   In this mode the private PyPI server is not published on the host; it is only reachable
   from containers inside the isolated network. Verify the isolation from within the
   container — external requests must fail while the private index stays reachable:

   .. code-block:: bash

      docker compose -f docker-compose.yml -f docker-compose.airgap.yml exec xinference \
         curl -s -m 5 https://pypi.org -o /dev/null || echo "external access blocked"
      docker compose -f docker-compose.yml -f docker-compose.airgap.yml exec xinference \
         curl -s http://xinference-pypiserver:8080/health

Smoke test
----------
.. code-block:: bash

   # The private index serves your wheels:
   curl http://localhost:8080/simple/

   # Inside the container, installs resolve against the private index only:
   docker compose exec xinference python3 -m pip config list
   docker compose exec xinference uv pip install --dry-run --python /usr/bin/python3 <some-package-in-wheels>

Then launch a model and watch ``docker compose logs -f xinference`` — package installation logs
should reference ``http://xinference-pypiserver:8080/simple``.
