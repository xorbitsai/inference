.. _launching_models:

================
Launching Models
================

You can launch models using Xinference's command line tool or client.

First, make sure you have installed the Xinference SDK:

.. code-block:: bash

   pip install xinference

Launching models with CLI
==========================================

.. note:: Suppose you have started the Xinference server endpoint at ``http://0.0.0.0:8001``. 


To launch an instance of Llama2 chat model:

.. code-block:: bash

   xinference launch --model-name "llama-2-chat" \
                     --model-format "ggmlv3" \
                     --size-in-billions 13 \
                     --endpoint "http://0.0.0.0:8001"

To launch an instance of General Text Embeddings (GTE) model:

.. code-block:: bash

   xinference launch --model-name "gte-base" \
                     --model-type "embedding" \
                     --endpoint  "http://0.0.0.0:8001"


Using RESTfulClient to Launch the Model
=======================================
.. code-block:: python

   from xinference.client import RESTfulClient
   c = RESTfulClient("http://0.0.0.0:8001")
   model_uid = client.launch_model(model_name="llama-2-chat", 
                                   model_format="ggmlv3",
                                   size_in_billions=13)

After launching the model, you can see the corresponding model's uid in the console. 
You will need to use this model's UID as a handle to interact with the model using the client. 
You can also list the running models using the ``list_models`` method:

.. code-block:: python

   from xinference.client import RESTfulClient
   client = RESTfulClient("http://127.0.0.1:8001")
   print(client.list_models())