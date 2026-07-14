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
* Get the whole ``xinference/deploy/docker`` directory. The compose file bind-mounts ``pip.conf`` and ``./wheels`` from the same directory, so downloading ``docker-compose.yml`` alone is not sufficient:

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

The ``offline`` compose profile solves this by starting a private PyPI server
(`pypiserver <https://github.com/pypiserver/pypiserver>`_) next to Xinference. It serves wheels
from the local ``./wheels`` directory, and the offline configuration points every runtime
``pip`` / ``uv`` invocation inside the Xinference container at it.

Step 1: Prepare wheels on an Internet-connected machine
-------------------------------------------------------
Download the wheels of every package your models declare in their ``virtualenv`` section
(shown on each model card in the Web UI, or in the built-in model JSON specs). The GPU image
runs Python 3.12 on ``x86_64``:

.. code-block:: bash

   python3 -m pip download \
      --dest ./wheels \
      --only-binary=:all: \
      --python-version 312 \
      --platform manylinux2014_x86_64 \
      'transformers>=4.53.3' accelerate

Copy the resulting ``wheels`` directory into ``xinference/deploy/docker/wheels`` on the offline
host. Also transfer the Docker images (``docker save`` / ``docker load``): the Xinference image
and ``pypiserver/pypiserver:v2.3.2``.

On Linux hosts, grant the pypiserver container user (UID 9898) access to the directory —
its entrypoint requires read/write/execute permission bits, although the volume itself is
mounted read-only:

.. code-block:: bash

   chmod -R a+rwX ./wheels

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

   Both pieces are required because they cover different code paths. ``pip.conf`` feeds
   Xinference's pip-config inheritance, which passes the private index explicitly to the
   per-model virtual-env installer; the ``UV_*`` variables in ``offline.env`` cover ``uv``
   invocations that do not carry index flags (such as the dependency-resolution dry-run).
   Removing either half breaks the offline install chain.

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

.. warning::

   When launching models with the **vLLM** or **SGLang** engines, Xinference by default resolves
   some dependencies from hardcoded public indexes (``wheels.vllm.ai``,
   ``download.pytorch.org``). The offline ``pip.conf`` above overrides them with the private
   index, which means those wheels must be present in ``./wheels`` — mirror the required
   ``vllm`` / ``torch`` CUDA wheels when preparing Step 1. Alternatively, set
   ``XINFERENCE_ENABLE_VIRTUAL_ENV=0`` in ``offline.env`` to skip runtime installs entirely and
   rely on the packages baked into the image.

Offline model weights
---------------------
The offline configuration sets ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``, so model
weights are read from the local cache only. Populate the cache before going offline, either by
pointing the cache variables in ``.env`` at host directories that already contain the models,
or by launching each model once on a connected host and copying the volumes. Models can also be
loaded from arbitrary local paths by registering custom models or passing ``--model-path``.

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
