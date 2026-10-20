===========================
The internals of Xinference
===========================

.. contents:: Table of contents:
   :local:

Overview
========
Xinference leverages `Xoscar <https://github.com/xorbitsai/xoscar>`_, an actor programming framework we designed,
as its core component to manage machines, devices, and model inference processes. Each actor serves as a basic
unit for model inference and various inference backends can be integrate into the actor, enabling us to support
multiple inference engines and hardware. These actors are hosted and scheduled within actor pools, which are
designed to be asynchronous and non-blocking and function as resource pools.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/actor.svg" style="background-color: transparent", width="77%">

====

Both supervisor and worker are actor instances. Initially, an actor pool, serving as a resource pool, needs to be created
on each server; and each actor can utilize a CPU core or a GPU device. Each server has its own address (IP address or
hostname), so actors on different computing nodes can communicate with each other through these addresses. See `Actor`_ for more information.

RESTful API
===========
The RESTful API is implemented using `FastAPI <https://github.com/tiangolo/fastapi>`_, as specified in
`api/restful_api.py <https://github.com/xorbitsai/inference/tree/main/xinference/api/restful_api.py>`_.

::

  self._router.add_api_route("/status", self.get_status, methods=["GET"])

This is an example of the API ``/status``, it's corresponding function is ``get_status``. You can add connection
between RESTful API and the backend function you want in `api/restful_api.py <https://github.com/xorbitsai/inference/tree/main/xinference/api/restful_api.py>`_.

Command Line
============
The Command Line is implemented using `Click <https://click.palletsprojects.com/>`_, as specified in
`deploy/cmdline.py <https://github.com/xorbitsai/inference/tree/main/xinference/deploy/cmdline.py>`_,
allowing users to interact with the Xinference deployment features directly from the terminal.

Entry Points
------------
Take the command-lines we implemented as examples:

- ``xinference``: Provides commands for model management, including registering/unregistering models, listing all
  registered/running models, and launching or terminating specific models.
  It also features interactive commands like generate and chat for testing and interacting with deployed models in real-time.

- ``xinference-local``: Starts a local Xinference service.

- ``xinference-supervisor``: Initiates a supervisor process that manages and monitors worker actors within a distributed setup.

- ``xinference-worker``: Starts a worker process that executes tasks assigned by the supervisor, utilizing available
  computational resources effectively.

Each command is equipped with ``options`` and ``flags`` to customize its behavior, such as specifying log levels,
host addresses, port numbers, and other relevant settings.

Python projects define command-line console entry points in ``pyproject.toml``.

::

  [project.scripts]
  xinference = "xinference.deploy.cmdline:cli"
  xinference-local = "xinference.deploy.cmdline:local"
  xinference-supervisor = "xinference.deploy.cmdline:supervisor"
  xinference-worker = "xinference.deploy.cmdline:worker"

The command-line ``xinference`` can be referred to code in ``xinference.deploy.cmdline:cli``.

Click
-----
We use Click to implement a specific command-line:

::

  @click.option(
        "--host",
        "-H",
        default=XINFERENCE_DEFAULT_DISTRIBUTED_HOST,
        type=str,
        help="Specify the host address for the supervisor.",
    )
    @click.option(
        "--port",
        "-p",
        default=XINFERENCE_DEFAULT_ENDPOINT_PORT,
        type=int,
        help="Specify the port number for the Xinference web ui and service.",
    )

For example, the ``xinference-local`` command allows you to define the host address and port.

Actor
=====
Xinference is fundamentally based on `Xoscar <https://github.com/xorbitsai/xoscar>`_, our actor framework,
which can manage computational resources and Python processes to support scalable and concurrent programming.
The following is a pseudocode demonstrating how our Worker Actor works, the actual Worker Actor is more complex than this.

::

  import xoscar as xo

  class WorkerActor(xo.Actor):
      def __init__(self, *args, **kwargs):
          ...
      async def launch_model(self, model_id, n_gpu, ...):
          # launch an inference engine, use specific model class to load model checkpoints
          ...
      async def list_models(self):
          # list models on this actor
          ...
      async def terminate_model(self, model_id):
          # terminate the model
          ...
      async def __post_create__(self):
          # called after the actor instance is created
          ...
      async def __pre_destroy__(self):
          # called before the actor instance is destroyed
          ...

We use the ``WorkerActor`` as an example to illustrate how we build the Xinference. Each actor class
is a standard Python class that inherits from ``xoscar.Actor``. An instance of this class is a specific actor
within the actor pool.

- **Define Actor Actions**: Each actor needs to define certain actions or behaviors to accomplish specific tasks.
  For instance, the model inference ``WorkerActor`` needs to launch the model (``launch_model``), list the models
  in this actor (``list_models``), terminate a model (``terminate_model``). There are two special methods worth
  noting. The ``__post_create__`` is invoked before the actor is created, allowing for necessary initializations.
  The ``__pre_destroy__`` is called after the actor is destroyed, allowing for cleanup or finalization tasks.

- **Reference Actor and Invoke Methods**: When an actor is created, it yields a reference variable so that other
  actors can reference it. The actor reference can also be referenced with the address. Suppose the ``WorkerActor``
  is created and the reference variable is ``worker_ref``,  the ``launch_model`` method of this actor class can
  be invoked by calling ``worker_ref.launch_model()``.
  Even if the actor's method is originally a synchronized method, when called with an actor reference, it will
  become as an asynchronous method.

- **Inference Engine**: The actor can manage the process, and the inference engine is also a process. In the launch
  model part of the ``WorkerActor``, we can initialize different inference engines according to the user's need.
  Therefore, Xinference can support multiple inference engines and can easily adapt to new inference engines in the
  future.

See `Xoscar document <https://xoscar.dev/en/latest/getting_started/llm-inference.html>`_ for more actor use cases.

Asynchronous Programming
========================

Both Xinference and Xoscar highly utilize asynchronous programming of ``asyncio``.
Asynchronous programming is a programming paradigm that does not block.
Instead, requests and function calls are issued and executed in the background
and results are returned in the future. This enables us to perform
activities concurrently.

If you're not familiar with Pythons's ``asyncio``, you can see more tutorials for help:

  - `Python Asyncio Tutorial <https://bbc.github.io/cloudfit-public-docs/asyncio/asyncio-part-1.html>`__

  - `Real Python's asyncio Tutorial <https://realpython.com/async-io-python/>`__

  - `Python Official Documentation <https://docs.python.org/3/library/asyncio.html>`__


Model
=====

Xinference supports different types of models including large language models (LLMs), image models, audio models, embedding models, etc.
All models are implemented in `model/ <https://github.com/xorbitsai/inference/tree/main/xinference/model>`_.

LLM
---

Take `model/llm/ <https://github.com/xorbitsai/inference/tree/main/xinference/model/llm>`_ for example, it focuses on
the management and instantiation of LLMs. It includes detailed implementations for loading, configuring,
and deploying LLMs.

We support many backends such as GGML, PyTorch, and vLLM. Our generated content is compatible with the format of OpenAI, supporting features such as streaming output and returning chat completion format (for chat models only).
Therefore, there is a lot of adaptation work to be done after the model generate content. These tasks are not difficult, but they do require some time. When writing this part of the code, please refer to the `OpenAI API documentation <https://platform.openai.com/docs/introduction>`_ and the documentation of various inference backends, and make the necessary adaptations.

JSON
----

In `model/llm/models/ <https://github.com/xorbitsai/inference/tree/main/xinference/model/llm/models>`_,
each model has a JSON file containing all its variants. Add or remove a model by adding or deleting its file. Files are discovered automatically in case-insensitive filename order; the order of records within each file is preserved.

::

  {
      "model_name": "llama-2-chat",
      "model_ability": ["chat"],
      "model_specs": [
          {
              "model_format": "ggmlv3",
              "model_size_in_billions": 70,
              "quantization": ["q8_0", ...],
              "model_id": "TheBloke/Llama-2-70B-Chat-GGML",
          },
          ...
      ],
      "prompt_style": {
          "style_name": "LLAMA2",
          "system_prompt": "<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n",
          "roles": ["[INST]", "[/INST]"],
          "stop_token_ids": [2],
          "stop": ["</s>"]
      }
  }

This is an example of how to define the Llama-2 chat model. The ``model_specs`` define the information of the model, as one model family
usually comes with various sizes, quantization methods, and file formats.
For instance, the ``model_format`` could be ``pytorch`` (using Hugging Face Transformers or vLLM as backend),
``ggmlv3`` (a tensor library associated with llama.cpp), or ``gptq`` (a post-training quantization framework).
The ``model_id`` defines the repository of the model hub from which Xinference downloads the checkpoint files.
Furthermore, due to distinct instruction-tuning processes, different model families have varying prompt styles.
The ``prompt_style`` in the JSON file specifies how to format prompts for this particular model.
For example, ``system_prompt`` and ``roles`` are used to specify the instructions and personality of the model.

Code Walkthrough
================

The main code is located in the `xinference/ <https://github.com/xorbitsai/inference/tree/main/xinference>`_:

- `api/ <https://github.com/xorbitsai/inference/tree/main/xinference/api>`_: `restful_api.py <https://github.com/xorbitsai/inference/tree/main/xinference/api/restful_api.py>`_
  is the core part that sets up and runs the RESTful APIs.
  It integrates an authentication service (the specific code is located in `oauth2/ <https://github.com/xorbitsai/inference/tree/main/xinference/api/oauth2>`_),
  as some or all endpointsrequire user authentication.

- `client/ <https://github.com/xorbitsai/inference/tree/main/xinference/client>`_: This is the client of Xinference.

  - `oscar/ <https://github.com/xorbitsai/inference/tree/main/xinference/client/oscar>`_ defines the Actor Client which acts as
    a client interface for interacting with models deployed in a Xinference cluster.

  - `restful/ <https://github.com/xorbitsai/inference/tree/main/xinference/client/restful>`_ implements a RESTful client for
    interacting with a Xinference service.

- `core/ <https://github.com/xorbitsai/inference/tree/main/xinference/core>`_: This is the core part of Xinference.

  - `metrics.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/metrics.py>`_ and
    `resource.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/resource.py>`_
    defines a set of tools for collecting and reporting metrics and the status of node resources, including model throughput,
    latency, the usage of CPU and GPU, memory usage, and more.

  - `worker.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/worker.py>`_ and
    `supervisor.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/supervisor.py>`_
    respectively define the logic for worker actors and supervisor actor. Worker actors are responsible for carrying out specific
    model computation tasks, while supervisor actors manage the lifecycle of worker nodes, schedule tasks, and monitor system states.

  - `status_guard.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/status_guard.py>`_ implements a status monitor
    to track the status of models (like creating, updating, terminating, etc.). It allows querying status information of model instances
    and managing these statuses based on the model's UID.

  - `cache_tracker.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/cache_tracker.py>`_ defines a cache tracker for
    recording and managing cache status and information of model versions. It supports recording cache locations and statuses of model
    versions and querying model version information based on model names.

  - `event.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/event.py>`_ defines an event collector for gathering and
    reporting various runtime events of models, such as information, warnings, and errors.
    `model.py <https://github.com/xorbitsai/inference/tree/main/xinference/core/model.py>`_ defines a Model Actor, the core component for
    direct model interactions. The Model Actor is responsible for executing model inference requests, handling input and output data streams,
    and supports various types of model operations.

- `deploy/ <https://github.com/xorbitsai/inference/tree/main/xinference/deploy>`_: It provides a command-line interface (CLI) for interacting
  with the Xinference framework, allowing users to perform operations by command line. See `Command Line`_ for more information.

- `locale/ <https://github.com/xorbitsai/inference/tree/main/xinference/locale>`_: It supports multi-language localization. By simply adding
  and updating JSON translation files, it becomes possible to support more languages, improving user experience.

- `model/ <https://github.com/xorbitsai/inference/tree/main/xinference/model>`_: It provides a structure for model descriptions, creation,
  and caching. See `Model`_ for more information.

- `frontend/ <https://github.com/xorbitsai/inference/tree/main/frontend>`_: The Next.js code of the frontend (Web UI).


Runtime extensions
==================

Set ``XINFERENCE_EXTENSIONS`` to comma-separated ``module:factory`` entries before starting Xinference. Each process loads its own extension instances, including spawned API, worker, and recovery processes. An unset variable preserves the standard runtime; an invalid configured extension fails explicitly.

A factory returns an object with a unique ``name``, ``api_version = 1``, and four methods: ``configure_api(context)``, ``worker_metadata()``, ``filter_worker_resources(worker, workers)``, and ``control_operation(operation, payload)``. API context includes the app, router, authentication service, authentication mode, and an asynchronous supervisor reference getter. Routes and middleware are composed before the static frontend fallback. Reuse the authentication service when protecting extension routes.

The supervisor verifies matching worker extensions and supplies the live worker inventory to resource filters. Filters may narrow visible GPU indices or raise to reject admission, including CPU launches. Allocation and process creation both check policy, covering preparation and recovery. Named control operations run on the supervisor; extensions must validate their operation names and payloads. The default frontend navigation configuration is empty and can be composed by a distribution build.

Factories and runtime hooks must be synchronous and non-blocking; awaitable results are rejected. API and actor startup load configured factories before serving requests or registering workers. Actor hooks run on the actor's event loop.

``context.auth_enabled`` describes server-wide authentication configuration, not the current request. Extension route handlers must annotate a ``starlette.responses.Response`` subclass as their return type, as required by the standard API.

Resource policies apply to model placement, allocation, and process creation, with a check before GPU cleanup. They do not filter cache/download-only worker placement.

Extensions may optionally implement synchronous, non-blocking ``model_catalog_credentials(request, model_type)``. For ``POST /v1/models/update_type``, this hook runs after the configured authentication and ``models:register`` permission checks. Return ``xinference.extensions.ModelCatalogCredentials(token=...)`` to supply a server-owned catalog token, or ``None`` to keep anonymous refresh. Existing extensions need no changes. Multiple credential providers and provider failures reject the refresh without an anonymous retry. Providers must keep tokens out of their own logs and HTTP error details or headers.

Credentials travel only with that refresh through the supervisor to each worker, which sends ``X-Models-Token`` to ``https://model.xinference.io/api/models/download``. Xinference does not persist the token or export it to the process environment; its credential object hides the token in actor argument logs. Credentialed requests reject redirects. Remote errors are sanitized, and unavailable or failing workers cause the refresh to fail. Workers that already updated are not rolled back; retry after correcting the failure. Update all runtime nodes before using this hook. Catalog updates affect the shared model registry; this hook does not provide per-user model visibility or model-weight download credentials.

To refresh one model, send ``POST /v1/models/update_spec`` with ``model_type`` and ``model_name``. It uses the same credential hook and ``models:register`` permission. The catalog response must contain one record or a non-empty list of variants, all with the requested ``model_name``. Only that model is replaced in the downloaded catalog; other downloaded models and packaged built-ins remain available. Full-category refresh through ``/v1/models/update_type`` is unchanged. Both paths coordinate file writes and publish complete JSON atomically. Upgrade the API, supervisor and all workers together before using the new endpoint.

Malformed request bodies return HTTP 400. Credential provider failures return HTTP 500; explicit HTTP errors from providers keep their status and headers. Worker refresh failures return HTTP 502. No available workers or an unreachable supervisor return HTTP 503.

For family-shaped world records, ``model_family`` is used when the top-level ``model_name`` is absent. Before a single-model update is published, the merged catalog must load successfully through the existing model-family loaders. Single-model downloads request the Hub README; a later full-category refresh replaces the catalog with the full-download response, which may omit README fields.

Advanced-auth storage
---------------------

Extensions may optionally implement ``create_auth_store(context)`` to select advanced-auth storage before the authentication service is initialized. ``AuthStoreContext`` supplies ``db_path`` (the requested default SQLite path) and ``purpose``: ``service``, ``reset_password`` or ``migrate``. Return an object implementing ``xinference.api.oauth2.advanced.auth_store.AuthStore``, or ``None`` to retain SQLite. At most one extension may provide this hook; multiple providers are rejected before either is called. Provider failures or invalid stores abort initialization without falling back to SQLite.

The store keeps the existing method signatures, row dictionaries and API-key name validation errors. It provides a mutable ``permission_resolver`` and a ``path`` property containing the local SQLite ``pathlib.Path`` or ``None`` for other storage. Setup, refresh-token rotation and password updates with token revocation must be atomic across processes; user deletion must preserve the existing dependent-record cleanup. Database drivers, schema upgrades and backend-specific configuration belong to the provider.

Provider stores shared by multiple API processes can set ``api_key_cache_enabled = False`` before service initialization. This bypasses the process-local API-key cache and reads the current key, owner status and model grants on each lookup. Omit the attribute or set it to ``True`` to retain caching; non-boolean values are rejected. Storage failures propagate instead of reusing earlier authorization.

The API service, offline administrator-password recovery and legacy JSON auth importer use the same selection mechanism. With a provider configured, ``--db-path`` is passed to its context; it does not override the provider's connection configuration. Use the same extension and backend configuration for the service and these tools. Unlike request/actor hooks, this synchronous initialization hook may connect to storage during startup. Store operations remain synchronous; asynchronous or incomplete implementations are rejected. Providers must keep connection credentials out of logs and error messages. Without a provider, SQLite behavior and dependencies are unchanged.
