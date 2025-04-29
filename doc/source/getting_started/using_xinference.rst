.. _using_xinference:

================
Using Xinference
================


Run Xinference Locally
======================

Let's start by running Xinference on a local machine and running a classic LLM model: ``qwen2.5-instruct``.

After this quickstart, you will move on to learning how to deploy Xinference in a cluster environment.

Start Local Server
------------------

First, please ensure that you have installed Xinference according to the instructions provided :ref:`here <installation>`.
To start a local instance of Xinference, run the following command:

.. tabs::

  .. tab:: shell

    .. code-block:: bash

        xinference-local --host 0.0.0.0 --port 9997

  .. tab:: output

    .. code-block:: bash

      INFO     Xinference supervisor 0.0.0.0:64570 started
      INFO     Xinference worker 0.0.0.0:64570 started
      INFO     Starting Xinference at endpoint: http://0.0.0.0:9997
      INFO     Uvicorn running on http://0.0.0.0:9997 (Press CTRL+C to quit)

.. note::
  By default, Xinference uses ``<HOME>/.xinference`` as home path to store necessary files such as logs and models,
  where ``<HOME>`` is the home path of current user.

  You can change this directory by configuring the environment variable ``XINFERENCE_HOME``.
  For example:

  .. code-block:: bash

    XINFERENCE_HOME=/tmp/xinference xinference-local --host 0.0.0.0 --port 9997

Congrats! You now have Xinference running on your local machine. Once Xinference is running, there are multiple ways
we can try it: via the web UI, via cURL, via the command line, or via the Xinference's python client.

You can visit the web UI at `http://127.0.0.1:9997/ui <http://127.0.0.1:9997/ui>`_ and visit `http://127.0.0.1:9997/docs <http://127.0.0.1:9997/docs>`_
to inspect the API docs.

You can install the Xinference command line tool and Python client using the following command:

.. code-block:: bash

   pip install xinference

The command line tool is ``xinference``. You can list the commands that can be used by running:

.. tabs::

  .. tab:: shell

    .. code-block:: bash

      xinference --help

  .. tab:: output

    .. code-block:: bash

      Usage: xinference [OPTIONS] COMMAND [ARGS]...

      Options:
        -v, --version       Show the version and exit.
        --log-level TEXT
        -H, --host TEXT
        -p, --port INTEGER
        --help              Show this message and exit.

      Commands:
        cached
        cal-model-mem
        chat
        engine
        generate
        launch
        list
        login
        register
        registrations
        remove-cache
        stop-cluster
        terminate
        unregister
        vllm-models


You can install the Xinference Python client with minimal dependencies using the following command.
Please ensure that the version of the client matches the version of the Xinference server.

.. code-block:: bash

   pip install xinference-client==${SERVER_VERSION}

.. _about_model_engine:

About Model Engine
------------------
Since ``v0.11.0`` , before launching the LLM model, you need to specify the inference engine you want to run.
Currently, xinference supports the following inference engines:

* ``vllm``
* ``sglang``
* ``llama.cpp``
* ``transformers``
* ``MLX``

About the details of these inference engine, please refer to :ref:`here <inference_backend>`.

Note that when launching a LLM model, the ``model_format`` and ``quantization`` of the model you want to launch
is closely related to the inference engine.

You can use ``xinference engine`` command to query the combination of parameters of the model you want to launch.
This will demonstrate under what conditions a model can run on which inference engines.

For example:

#. I would like to query about which inference engines the ``qwen-chat`` model can run on, and what are their respective parameters.

.. code-block:: bash

    xinference engine -e <xinference_endpoint> --model-name qwen-chat

#. I want to run ``qwen-chat`` with ``VLLM`` as the inference engine, but I don't know how to configure the other parameters.

.. code-block:: bash

    xinference engine -e <xinference_endpoint> --model-name qwen-chat --model-engine vllm

#. I want to launch the ``qwen-chat`` model in the ``GGUF`` format, and I need to know how to configure the remaining parameters.

.. code-block:: bash

    xinference engine -e <xinference_endpoint> --model-name qwen-chat -f ggufv2


In summary, compared to previous versions, when launching LLM models,
you need to additionally pass the ``model_engine`` parameter.
You can retrieve information about the supported inference engines and their related parameter combinations
through the ``xinference engine`` command.

.. note::

    Here are some recommendations on when to use which engine:

    - **Linux**

       - When possible, prioritize using **vLLM** or **SGLang** for better performance.
       - If resources are limited, consider using **llama.cpp**, as it offers more quantization options.
       - For other cases, consider using **Transformers**, which supports nearly all models.

    - **Windows**

       - It is recommended to use **WSL**, and in this case, follow the same choices as Linux.
       - Otherwise, prefer **llama.cpp**, and for unsupported models, opt for **Transformers**.

    - **Mac**

       - If supported by the model, use the **MLX engine**, as it delivers the best performance.
       - For other cases, prefer **llama.cpp**, and for unsupported models, choose **Transformers**.


Run qwen2.5-instruct
--------------------

Let's start by running a built-in model: ``qwen2.5-instruct``. When you start a model for the first time, Xinference will
download the model parameters from HuggingFace, which might take a few minutes depending on the size of the model weights.
We cache the model files locally, so there's no need to redownload them for subsequent starts.

.. note::
  Xinference also allows you to download models from other sites. You can do this by setting an environment variable
  when launching Xinference. For example, if you want to download models from `modelscope <https://modelscope.cn>`_,
  do the following:

  .. code-block:: bash

    XINFERENCE_MODEL_SRC=modelscope xinference-local --host 0.0.0.0 --port 9997

We can specify the model's UID using the ``--model-uid`` or ``-u`` flag. If not specified, Xinference will generate a unique ID.
The default unique ID will be identical to the model name.

.. tabs::

  .. code-tab:: bash shell

    xinference launch --model-engine <inference_engine> -n qwen2.5-instruct -s 0_5 -f pytorch

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://127.0.0.1:9997/v1/models' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model_engine": "<inference_engine>",
      "model_name": "qwen2.5-instruct",
      "model_format": "pytorch",
      "size_in_billions": "0_5"
    }'

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    model_uid = client.launch_model(
      model_engine="<inference_engine>",
      model_name="qwen2.5-instruct",
      model_format="pytorch",
      size_in_billions="0_5"
    )
    print('Model uid: ' + model_uid)

  .. code-tab:: bash output

    Model uid: qwen2.5-instruct

.. note::
  For some engines, such as vllm, users need to specify the engine-related parameters when
  running models. In this case, you can directly specify the parameter name and value in the
  command line, for example:

  .. code-block:: bash

    xinference launch --model-engine vllm -n qwen2.5-instruct -s 0_5 -f pytorch --gpu_memory_utilization 0.9

  `gpu_memory_utilization=0.9` will pass to vllm when launching model.

Congrats! You now have ``qwen2.5-instruct`` running by Xinference. Once the model is running, we can try it out either via cURL,
or via Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://127.0.0.1:9997/v1/chat/completions' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "qwen2.5-instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the largest animal?"
            }
        ]
      }'

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    model = client.get_model("qwen2.5-instruct")
    model.chat(
        messages=[
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    )

  .. code-tab:: json output

    {
      "id": "chatcmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
      "model": "qwen2.5-instruct",
      "object": "chat.completion",
      "created": 1688919187,
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The largest animal that has been scientifically measured is the blue whale, which has a maximum length of around 23 meters (75 feet) for adult animals and can weigh up to 150,000 pounds (68,000 kg). However, it is important to note that this is just an estimate and that the largest animal known to science may be larger still. Some scientists believe that the largest animals may not have a clear \"size\" in the same way that humans do, as their size can vary depending on the environment and the stage of their life."
          },
          "finish_reason": "None"
        }
      ],
      "usage": {
        "prompt_tokens": -1,
        "completion_tokens": -1,
        "total_tokens": -1
      }
    }

Xinference provides OpenAI-compatible APIs for its supported models, so you can use Xinference as a local drop-in replacement for OpenAI APIs. For example:

.. code-block:: python

  from openai import OpenAI
  client = OpenAI(base_url="http://127.0.0.1:9997/v1", api_key="not used actually")

  response = client.chat.completions.create(
      model="qwen2.5-instruct",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the largest animal?"}
      ]
  )
  print(response)

The following OpenAI APIs are supported:

- Chat Completions: `https://platform.openai.com/docs/api-reference/chat <https://platform.openai.com/docs/api-reference/chat>`_

- Completions: `https://platform.openai.com/docs/api-reference/completions <https://platform.openai.com/docs/api-reference/completions>`_

- Embeddings: `https://platform.openai.com/docs/api-reference/embeddings <https://platform.openai.com/docs/api-reference/embeddings>`_

Manage Models
-------------

In addition to launching models, Xinference offers various ways to manage the entire lifecycle of models.
You can manage models in Xinference through the command line, cURL, or Xinference's python client.

You can list all models of a certain type that are available to launch in Xinference:

.. tabs::

  .. code-tab:: bash shell

    xinference registrations -t LLM

  .. code-tab:: bash cURL

    curl http://127.0.0.1:9997/v1/model_registrations/LLM

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    print(client.list_model_registrations(model_type='LLM'))

The following command gives you the currently running models in Xinference:

.. tabs::

  .. code-tab:: bash shell

    xinference list

  .. code-tab:: bash cURL

    curl http://127.0.0.1:9997/v1/models

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    print(client.list_models())

When you no longer need a model that is currently running, you can remove it in the following way to free up the resources it occupies:

.. tabs::

  .. code-tab:: bash shell

    xinference terminate --model-uid "qwen2.5-instruct"

  .. code-tab:: bash cURL

    curl -X DELETE http://127.0.0.1:9997/v1/models/qwen2.5-instruct

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    client.terminate_model(model_uid="qwen2.5-instruct")

.. _distributed_getting_started:

Deploy Xinference In a Cluster
==============================

To deploy Xinference in a cluster, you need to start a Xinference supervisor on one server and Xinference workers
on the other servers.

First, make sure you have already installed Xinference on each of the servers according to the instructions
provided :ref:`here <installation>`. Then follow the steps below:

Start the Supervisor
--------------------
On the server where you want to run the Xinference supervisor, run the following command:

.. code-block:: bash

  xinference-supervisor -H "${supervisor_host}"

Replace ``${supervisor_host}`` with the actual host of your supervisor server.


You can the supervisor's web UI at `http://${supervisor_host}:9997/ui <http://${supervisor_host}:9997/ui>`_ and visit
`http://${supervisor_host}:9997/docs <http://${supervisor_host}:9997/docs>`_ to inspect the API docs.

Start the Workers
-----------------

On each of the other servers where you want to run Xinference workers, run the following command:

.. code-block:: bash

  xinference-worker -e "http://${supervisor_host}:9997" -H "${worker_host}"

.. note::
    Note that you must replace ``${worker_host}``  with the actual host of your worker server.

.. note::
  Note that if you need to interact with the Xinference in a cluster via the command line,
  you should include the ``-e`` or ``--endpoint`` flag to specify the supervisor server's endpoint. For example:

  .. code-block:: bash

      xinference launch -n qwen2.5-instruct -s 0_5 -f pytorch -e "http://${supervisor_host}:9997"

Using Xinference With Docker
=============================

To start Xinference in a Docker container, run the following command:

Run On Nvidia GPU Host
-----------------------

.. code-block:: bash

  docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 --gpus all xprobe/xinference:<your_version> xinference-local -H 0.0.0.0 --log-level debug

Run On CPU Only Host
-----------------------

.. code-block:: bash

  docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 xprobe/xinference:<your_version>-cpu xinference-local -H 0.0.0.0 --log-level debug

Replace ``<your_version>`` with Xinference versions, e.g. ``v0.10.3``, ``latest`` can be used for the latest version.

For more docker usage, refer to :ref:`Using Docker Image <using_docker_image>`.


What's Next?
============

Congratulations on getting started with Xinference! To help you navigate and make the most out of this
powerful tool, here are some resources and guides:

* :ref:`How to Use Client APIs for Different Types of Models <user_guide_client_api>`

* :ref:`Choosing the Right Backends for Your Needs <user_guide_backends>`
