.. _models_builtin_index:

==============
Builtin Models
==============


Xinference offers an extensive array of AI models, encompassing everything from text generation and multimodal models, to text embedding and rerank models.


List the Built-in Models
============================

You can list all models of a certain type that are available to launch in Xinference:

.. tabs::

  .. code-tab:: bash shell

    xinference registrations -t <model_type>

  .. code-tab:: bash cURL

    curl http://127.0.0.1:9997/v1/model_registrations/<model_type>

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    print(client.list_model_registrations(model_type='<model_type>'))

The following ``model_type`` is supported by Xinference:

* ``LLM``   
* ``multimodal``
* ``embedding``
* ``image`` 
* ``rerank``


.. toctree::
   :maxdepth: 1

   llm/index
   embedding/index
   rerank/index