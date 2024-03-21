========================================
The overall code framework of Xinference
========================================

.. contents:: Table of contents:
   :local:

Overview
--------
The main code is located in the ``xinference/``: 

- ``api/``: ``restful_api.py`` is the core file that sets up and runs the RESTful API, interfacing with various functionalities 
  like creating completions (presumably for LLMs), handling embeddings, reranking, and processing images, among others. 
  It also handles registration and deregistration of models, suggesting dynamic management of available machine learning models.
  The API supports operations like generating text completions, embeddings, reranking documents, and processing images. 
  It integrates an authentication service (the specific code is located in `oauth2/`), indicating that some or all endpoints require user authentication.
  See `RESTful API`_ for more information.

- ``client/``: This is the client for the client-server architecture comprehensive framework, supporting both synchronous
  and asynchronous operations. It also can handle streaming responses from models. This function is critical for processing
  real-time, streaming outputs from models. 
  
  - ``oscar/`` defines the Actor Client which acts as a client interface for interacting
    with models deployed in a server environment. It includes functionalities to register/unregister models, launch/terminate models,
    and interact with different types of models. This part heavily utilizes ``asyncio`` for asynchronous operations. 
  
  - ``restful/`` implements a RESTful client for interacting with a server that
    hosts machine learning models. It supports operations like listing models, launching and terminating models, and interacting
    with various types of models through HTTP requests.

- ``core/``: This is the core part of the Xinference. 
  
  - ``metrics.py`` and ``resource.py`` defines a set of tools for collecting and reporting metrics and the status of node resources, 
    including model throughput, latency, the usage of CPU and GPU, memory usage, and more.
  
  - ``image_interface.py`` and ``chat_interface.py`` implement Gradio interfaces for image and chat models, respectively. 
    These interfaces allow users to interact with models through a Web UI, such as generating images or engaging in chat. 
    They build user interfaces using the gradio package and communicate with backend models through RESTful APIs.
  
  - ``worker.py`` and ``supervisor.py`` respectively define the logic for worker nodes and supervisor nodes. Worker nodes are responsible
    for carrying out specific model computation tasks, while supervisor nodes manage the lifecycle of worker nodes, schedule tasks, and
    monitor system states.
  
  - ``status_guard.py`` implements a status monitor to track the status of models (like creating, updating, terminating, etc.). 
    It allows querying status information of model instances and managing these statuses based on the model's UID.
  
  - ``cache_tracker.py`` defines a cache tracker for recording and managing cache status and information of model versions. 
    It supports recording cache locations and statuses of model versions and querying model version information based on model names.

  - ``event.py`` defines an event collector for gathering and reporting various runtime events of models, such as information, warnings,
    and errors. ``model.py`` defines a Model Actor, the core component for direct model interactions. The Model Actor is responsible for
    executing model inference requests, handling input and output data streams, and supports various types of model operations.
    These two parts are all utilize `Xoscar <https://github.com/xorbitsai/xoscar>`_ for concurrent and distributed execution.

- ``deploy/``: It provides a command-line interface (CLI) for interacting with the Xinference framework, allowing users to perform
  operations by command line. See `Command Line`_ for more information.

- ``locale``: It supports multi-language localization. By simply adding and updating JSON translation files, it becomes possible to
  support more languages, improving user experience.

- ``model/``: It provides a structure for model descriptions, creation, and caching. It supports different types of models including
  large language models (LLMs), image models, audio models, embedding models, and rerank models. Take ``llm/`` for example, it focuses on
  the management and instantiation of large language models (LLMs). It includes detailed implementations for loading, configuring,
  and deploying LLMs, including handling different types of quantization and model formats. In ``llm/``, it supports many backends such as
  GGML, PyTorch, SGLang and vLLM.

- ``thirdparty/``: A thirdparty framework LLaVA (Large Language and Vision Assistant). It has the capable of understanding and generating
  responses that consider both the textual and visual context, useful for applications such as chatbots, image captioning, and enhanced
  language models that can interpret visual information. 

- ``web/ui/``: The js code of the frontend (Web UI).

Command Line
------------
::

  console_scripts =
      xinference = xinference.deploy.cmdline:cli
      xinference-local = xinference.deploy.cmdline:local
      xinference-supervisor = xinference.deploy.cmdline:supervisor
      xinference-worker = xinference.deploy.cmdline:worker


RESTful API
-----------
::

  /status
  /v1/models/prompts
  /v1/models/families
  /v1/models/vllm-supported
  /v1/cluster/info
  /v1/cluster/version
  /v1/cluster/devices
  /v1/address
  ...

Actor
-----
.. raw:: html

    <img class="align-center" alt="actor" src="../_static/actor.svg" style="background-color: transparent", width="77%">

====

Both supervisor and worker are actor instances. Initially, an actor pool, serving as a resource pool, needs to be created
on each server; and each actor can utilize a CPU core or a GPU device. Each server has its own address (IP address or
hostname), so actors on different computing nodes can communicate with each other through these addresses.

WorkerActor
^^^^^^^^^^^
The worker is the actual place for model serving. 
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

We use the ``WorkerActor`` as an example to illustrate how we build the model inference library. Each actor class
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

- **Inference Engine**: The actor can manage the process, and the inference engine is also a process. In the launch
  model part of the ``WorkerActor``, we can initialize different inference engines according to the user's need.
  Therefore, Xinference can support multiple inference engines and can easily adapt to new inference engines in the
  future.

Concurrency and Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Asynchronous I/O**: Our actor framework is designed in an asynchronous, non-blocking manner, enabling it to 
  handle data-intensive workloads. Large model inference is time-consuming, and traditional blocking calls often
  result in wasted time waiting for results to return. To address this, we have extensively used the philosophy
  of coroutine, such as Pythons's ``asyncio``, in our internal implementation. We treat the model inference task
  as an asynchronous task: we push the task into the pool when the request arrives and pull the task when computing
  resources are available.

- **Scheduling**: Our actor design is adept at managing concurrent requests and multiple model instances. Requests are
  dispatched to our per-model scheduler. Xinference retrieves the available actor from the actor pools and invokes the
  corresponding actor function to generate content. This per-model scheduler enables us to support one model with 
  multiple replicas or multiple models.

License
-------
`Apache 2 <https://github.com/xorbitsai/inference/blob/main/LICENSE>`_
