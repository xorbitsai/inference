.. _using_xinference:

================
Using Xinference
================


Run Xinference Locally
======================

Let's start by running Xinference on a local machine and running a classic LLM model: ``llama-2-chat``.

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
        chat
        generate
        launch
        list
        register
        registrations
        terminate
        unregister


You can install the Xinference Python client with minimal dependencies using the following command.
Please ensure that the version of the client matches the version of the Xinference server.

.. code-block:: bash

   pip install xinference-client==${SERVER_VERSION}

Run Llama-2
-----------

Let's start by running a built-in model: ``llama-2-chat``. When you start a model for the first time, Xinference will
download the model parameters from HuggingFace, which might take a few minutes depending on the size of the model weights.
We cache the model files locally, so there's no need to redownload them for subsequent starts.

.. note::
  Xinference also allows you to download models from other sites. You can do this by setting an environment variable
  when launching Xinference. For example, if you want to download models from `modelscope <https://modelscope.cn>`_,
  do the following:

  .. code-block:: bash

    export XINFERENCE_MODEL_SRC=modelscope xinference-local --host 0.0.0.0 --port 9997

We can specify the model's UID using the ``--model-uid`` or ``-u`` flag. If not specified, Xinference will generate a random ID.
This create a new model instance with unique ID ``my-llama-2``:

.. tabs::

  .. code-tab:: bash shell

    xinference launch -u my-llama-2 -n llama-2-chat -s 13 -f pytorch

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://127.0.0.1:9997/v1/models' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model_uid": "my-llama-2",
      "model_name": "llama-2-chat",
      "model_format": "pytorch",
      "size_in_billions": 13
    }'

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    model_uid = client.launch_model(
      model_uid="my-llama-2",
      model_name="llama-2-chat",
      model_format="pytorch",
      size_in_billions=13
    )
    print('Model uid: ' + model_uid)

  .. code-tab:: bash output

    Model uid: my-llama-2

Congrats! You now have ``llama-2-chat`` running by Xinference. Once the model is running, we can try it out either command line, via cURL,
or via Xinference's python client:

.. tabs::

  .. code-tab:: bash shell

    xinference chat --model-uid my-llama-2
    User: What is the largest animal?

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://127.0.0.1:9997/v1/chat/completions' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "my-llama-2",
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
    model = client.get_model("my-llama-2")
    print(model.chat(
        prompt="What is the largest animal?",
        system_prompt="You are a helpful assistant.",
        chat_history=[]
    ))

  .. code-tab:: json output

    {
      "id": "chatcmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
      "model": "my-llama-2",
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
      model="my-llama-2",
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

    xinference terminate --model-uid "my-llama-2"

  .. code-tab:: bash cURL

    curl -X DELETE http://127.0.0.1:9997/v1/models/my-llama-2

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    client.terminate_model(model_uid="my-llama-2")

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

  xinference-worker -e "http://${supervisor_host}:9997"


.. note::
  Note that if you need to interact with the Xinference in a cluster via the command line,
  you should include the ``-e`` or ``--endpoint`` flag to specify the supervisor server's endpoint. For example:

  .. code-block:: bash

      xinference launch -n llama-2-chat -s 13 -f pytorch -e "http://${supervisor_host}:9997"

What's Next?
============

Congratulations on getting started with Xinference! To help you navigate and make the most out of this
powerful tool, here are some resources and guides:

* :ref:`How to Use Client APIs for Different Types of Models <user_guide_client_api>`

* :ref:`Choosing the Right Backends for Your Needs <user_guide_backends>`
