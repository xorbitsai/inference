.. _index:

.. raw:: html

    <img class="align-center" alt="Xinference Logo" src="../_static/favicon.svg" style="background-color: transparent", width="180px">

====

=======================================================
Xinference: A Platform for Managing Large Model Serving
=======================================================

Introduction
------------
Xinference is an inference framework designed to simplify model serving. It provides the following features:

- **User-Friendly Interface**: Xinference simplifies interactions through a Web UI. It also provides
  command line tools and OpenAI-compatible APIs, integrating seamlessly with widely used libraries
  such as LangChain, LlamaIndex, or Dify. This ease of access allows users to effortlessly manage
  the lifecycle of the latest models, from download to deployment.

- **Seamless Backend Integration and Deployment**: Integrating to a variety of inference engines
  (like PyTorch, vLLM, llama.cpp, TensorRT-LLM) and supporting diverse hardware, Xinference abstracts
  the complexity of backend configurations. It enables the efficient distribution of model inference
  tasks across different devices and servers, providing a unified API endpoint for streamlined user access.

- **Diverse Model Support**: Embracing a wide range of cutting-edge models, including those for pre-training,
  chat, embedding, and multimodal tasks, Xinference is tailored to accommodate different model requirements
  and configurations, such as system prompts. This versatility ensures users can focus on their core objectives
  without getting bogged down by the intricacies of model management.

Xinference leverages `Xoscar <https://github.com/xorbitsai/xoscar>`_, an actor programming framework we designed, 
as its core component to manage machines, devices, and model inference processes. Each actor serves as a basic
unit for model inference and various inference engines can be integrate into the actor, enabling us to support 
multiple inference backends and hardware. These actors are hosted and scheduled within actor pools, which are
designed to be asynchronous and non-blocking and function as resource pools.

Mangaging Model Serving with Xinference
---------------------------------------

Using Xinference
^^^^^^^^^^^^^^^^
Xinference can be used either on users’ private on-premise clouds or on public clouds:

- **Launching Xinference Cluster in Private Cloud**: After being installed by pip or Docker, users can launch
  Xinference either on a local machine or a cluster. Users can use the commands to start a Xinference cluster:
  ::

    # on the supervisor server
    xinference-supervisor -H '${sv_host}'

    # on the worker server
    xinference-worker -e 'http://${sv_host}:9997'
  
  Users should first launch the supervisor, and start workers on other servers. The supervisor is responsible
  for coordination, whereas the worker manages the available resources (i.e., CPUs or GPUs) and executes the
  inference requests. The workers establish connections to the supervisor using the hostname and port, thereby
  setting up a Xinference cluster. In the local mode, both the supervisor and worker are launched on the same
  local computer.

- **Xinference Services on Public Cloud**: We also support public clouds, where computing resources can be auto-scaled.

Lifecycle of Model Serving
^^^^^^^^^^^^^^^^^^^^^^^^^^
Xinference's lifecycle of model serving is centered around models, primarily including managing models (using built-in
open-source models or registering custom models), launching a model, listing running models, using a model, monitoring,
and terminating running models.

- **Supported Models**: At present, our platform supports a variety of models, including chat, generate, embedding,
  rerank, text-to-image, and image-to-image. Xinference offers built-in models and also allows users to register
  their own custom models. The built-in models, such as Llama and Gemma, are open-source and utilize checkpoint
  files downloaded directly from model hubs, without any fine-tuning. If users fine-tune an open-source model,
  they have the option to register it with Xinference.

- **Supported Inference Engines**: Currently, we have integrated PyTorch (the Hugging Face's Transformers), vLLM,
  llama.cpp, and TensorRT-LLM as our inference backends.

- **Launch a Model**: Once a user has selected a model and configured necessary parameters, they can initiate it.
  The model will be assigned to a worker within the Xinference cluster and an inference engine will be launched
  on that worker. 

- **List Running Models**: Users can launch one model with multiple replicas or multiple different models. After
  initiation, the model will possess a unique model ID, which will be used to index the model for subsequent use
  and management.

- **Using a Model**: Users can use a model by using the Web UI, or the RESTful API, which is compatible with the
  OpenAI developer platform. The RESTful API allows Xinference to be used as a drop-in replacement for OpenAI.

- **Cluster and Model Monitoring**: Once the Xinference service is accessible to users, it exports metrics related
  to the cluster and model for monitoring. The cluster metrics include the available and used GPU memory. The model
  metrics comprise the count of received requests and sent responses, throughput(measured in tokens/s), and the first
  token latency in milliseconds.

- **Terminate Running Models**: If a model is no longer required, users can terminate it, which will release the
  corresponding computational resources.

User Interface
^^^^^^^^^^^^^^
Currently, we offer three types of interfaces to our users: Web UI, RESTful API, and command line:

- **Web UI**: Users can access the Web UI in their browser. The entire life cycle of the model serving can be
  completed on the graphical user interfaces. This type of interface is suitable for beginners with limited
  technical knowledge. 

- **RESTful API**: Users can also get access to models via our RESTful API using a development toolkit like Python,
  Node.js, or curl. Users can easily migrate from OpenAI to Xinference. The RESTful API is aimed at advanced users
  who need to interact with models programmatically.

- **Command Line**: Users can also interact with Xinference on the node where the supervisor is located using command
  lines such as ``xinference launch`` for launching a model, and ``xinference terminate`` for shutting down a model.

Design and Implementation
-------------------------
Xinference is fundamentally based on Xoscar, our actor framework, which can manage computational resources and Python
processes to support scalable and concurrent programming. This section will first introduce our actor framework,
followed by an explanation of how Xinference is developed based on this actor framework.

System Overview
^^^^^^^^^^^^^^^
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

Model Management
^^^^^^^^^^^^^^^^
For the inference engine management part, we have written modular code that includes loading models, formatting prompts,
and stopping when encountering end-of-sequence (EOS) tokens. Different models can reuse these codes. We utilize JSON files
to manage the metadata of emerging open-source models. Adding a new model does not necessitate writing new code; it merely
requires appending new metadata to the existing JSON file.

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

This is an example of how to define the Llama-2 chat model. ``model_specs`` define the information of the model, as one model
family usually comes with various sizes, quantization methods, and file formats. For instance, the ``model_format`` could be
``pytorch`` (using Hugging Face Transformers or vLLM as backend), ``ggmlv3`` (a tensor library associated with llama.cpp), or
``gptq`` (a post-training quantization framework). The ``model_id`` defines the repository of the model hub from which
Xinference downloads the checkpoint files. Furthermore, due to distinct instruction-tuning processes, different model families
have varying prompt styles. The ``prompt_style`` in the JSON file specifies how to format prompts for this particular model.
For example, ``system_prompt`` and ``roles`` are used to specify the instructions and personality of the model.

The current JSON format also supports the registration of custom models; custom model information is stored according to the
aforementioned fields. Moreover, the definitions of other models (e.g., embedding model and multimodal) are quite similar,
with fields slightly different.

License
-------
`Apache 2 <https://github.com/xorbitsai/inference/blob/main/LICENSE>`_
