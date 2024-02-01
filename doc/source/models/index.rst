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
                             --endpoint "http://<XINFERENCE_HOST>:<XINFERENCE_PORT>" \

  .. code-tab:: bash cURL

    curl http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/model_registrations/<MODEL_TYPE>

  .. code-tab:: python

    from xinference.client import Client
    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    print(client.list_model_registrations(model_type='<MODEL_TYPE>'))

The following ``MODEL_TYPE`` is supported by Xinference:

.. list-table::
   :widths: 25 50 50
   :header-rows: 1

   * - Type
     - Description
     - Index

   * - LLM
     - Text generation models or large language models
     - :ref:`Index <models_llm_index>`

   * - embedding
     - Text embeddings models
     - :ref:`Index <models_embedding_index>`

   * - image
     - Image generation or manipulation models
     - :ref:`Index <models_image_index>`

   * - audio
     - Audio models
     - :ref:`Index <models_audio_index>`

   * - rerank
     - Rerank models
     - :ref:`Index <models_rerank_index>`


You can see all the built-in models supported by xinference :ref:`here <models_builtin_index>`. If the model 
you need is not available, xinference also allows you to register your own :ref:`custom models <models_custom>`.

Launch Model
============================

You can launch a model in Xinference either via command line or Xinference's Python client:

.. tabs::

  .. code-tab:: bash shell

    xinference launch --model-name <MODEL_NAME> \
                      --model-type <MODEL_TYPE> \
                      --endpoint "http://<XINFERENCE_HOST>:<XINFERENCE_PORT>" \


  .. code-tab:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model_uid = client.launch_model(model_name="<MODEL_NAME>", model_type="<MODEL_TYPE>")
    print(model_uid)


For model type ``LLM``, launching the model requires not only specifying the model name, but also the size of the parameters
and the model format.  Please refer to the list of LLM :ref:`model families <models_llm_index>`.


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
