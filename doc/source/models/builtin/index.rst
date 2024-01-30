.. _models_builtin_index:

==============
Builtin Models
==============


Xinference offers an extensive array of AI models, encompassing everything from text generation and multimodal models, 
to text embedding and rerank models.


List the Built-in Models
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

* ``LLM``   
* ``embedding``
* ``image`` 
* ``rerank``


Launch a Built-in Model
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


.. toctree::
   :maxdepth: 1

   llm/index
   embedding/index
   image/index
   rerank/index
