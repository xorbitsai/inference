.. _models_index:

======
Models
======

List Models
============================

You can list all models of a certain type that are available to launch in Xinference:

.. tabs::

  .. code-tab:: bash shell

    xinference registrations --model-type <MODEL_TYPE> \
                             [--endpoint "http://<XINFERENCE_HOST>:<XINFERENCE_PORT>"] \

  .. code-tab:: bash cURL

    curl http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/model_registrations/<MODEL_TYPE>

  .. code-tab:: python

    from xinference.client import Client
    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    print(client.list_model_registrations(model_type='<MODEL_TYPE>'))

The following ``MODEL_TYPE`` is supported by Xinference:

.. grid:: 2

    .. grid-item-card::  LLM
      :link: models_llm_index
      :link-type: ref

      Text generation models or large language models

    .. grid-item-card::  embedding
      :link: models_embedding_index
      :link-type: ref

      Text embeddings models

.. grid:: 2

    .. grid-item-card::  image
      :link: models_image_index
      :link-type: ref

      Image generation or manipulation models

    .. grid-item-card::  audio
      :link: models_audio_index
      :link-type: ref

      Audio models

.. grid:: 2

    .. grid-item-card::  rerank
      :link: models_rerank_index
      :link-type: ref

      Rerank models



You can see all the built-in models supported by xinference :ref:`here <models_builtin_index>`. If the model 
you need is not available, Xinference also allows you to register your own :ref:`custom models <models_custom>`.


Launch and Terminate Model
============================

Each running model instance will be assigned a unique model uid. By default, the model uid is equal to the model name.
This unique id can be used as a handle for the further usage. You can manually assign it by passing ``--model-uid`` option
in the launch command. 

You can launch a model in Xinference either via command line or Xinference's Python client:

.. tabs::

  .. code-tab:: bash shell

    xinference launch --model-name <MODEL_NAME> \
                      [--model-engine <MODEL_ENGINE>] \
                      [--model-type <MODEL_TYPE>] \
                      [--model-uid <MODEL_UID>] \
                      [--endpoint "http://<XINFERENCE_HOST>:<XINFERENCE_PORT>"] \


  .. code-tab:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model_uid = client.launch_model(
      model_name="<MODEL_NAME>",
      model_engine="<MODEL_ENGINE>",
      model_type="<MODEL_TYPE>"
      model_uid="<MODEL_UID>"
    )
    print(model_uid)


For model type ``LLM``, launching the model requires not only specifying the model name, but also the size of the parameters
, the model format and the model engine.  Please refer to the list of LLM :ref:`model families <models_llm_index>`.

The following command gives you the currently running models in Xinference:

.. tabs::

  .. code-tab:: bash shell

    xinference list [--endpoint "http://<XINFERENCE_HOST>:<XINFERENCE_PORT>"]


  .. code-tab:: bash cURL

    curl http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/models


  .. code-tab:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    print(client.list_models())

When you no longer need a model that is currently running, you can remove it in the
following way to free up the resources it occupies:


.. tabs::

  .. code-tab:: bash shell

    xinference terminate --model-uid "<MODEL_UID>" [--endpoint "http://<XINFERENCE_HOST>:<XINFERENCE_PORT>"]

  .. code-tab:: bash cURL

    curl -X DELETE http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/models/<MODEL_UID>


  .. code-tab:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    client.terminate_model(model_uid="<MODEL_UID>")


Model Usage
============================


.. grid:: 2

    .. grid-item-card::  Chat & Generate
      :link: chat
      :link-type: ref

      Learn how to chat with LLMs in Xinference.

    .. grid-item-card::  Tools
      :link: tools
      :link-type: ref

      Learn how to connect LLM with external tools.


.. grid:: 2

    .. grid-item-card::  Embeddings
      :link: embed
      :link-type: ref

      Learn how to create text embeddings in Xinference.

    .. grid-item-card::  Rerank
      :link: rerank
      :link-type: ref

      Learn how to use rerank models in Xinference.


.. grid:: 2

    .. grid-item-card::  Images
      :link: image
      :link-type: ref

      Learn how to generate images with Xinference.

    .. grid-item-card::  Vision
      :link: vision
      :link-type: ref

      Learn how to process image with LLMs.


.. grid:: 2

    .. grid-item-card::  Audio
      :link: audio
      :link-type: ref

      Learn how to turn audio into text or text into audio with Xinference.


.. toctree::
   :maxdepth: 2

   model_abilities/index
   builtin/index
   custom
   sources/sources
   lora
   model_memory
