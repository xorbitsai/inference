.. _models_custom:

============================
Custom Models (Experimental)
============================

Custom models are currently an experimental feature and are expected to be officially released in
version v0.2.0.

Custom models from Hugging Face
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom model based on the following template:

.. code-block:: python

   custom_model = {
     "version": 1,
     # model name. must start with a letter or a
     # digit, and can only contain letters, digits,
     # underscores, or dashes.
     "model_name": "nsql-2B",
     # supported languages
     "model_lang": [
       "en"
     ],
     # model abilities. could be "embed", "generate"
     # and "chat".
     "model_ability": [
       "generate"
     ],
     # model specifications.
     "model_specs": [
       {
         # model format.
         "model_format": "pytorch",
         "model_size_in_billions": 2,
         # quantizations.
         "quantizations": [
           "4-bit",
           "8-bit",
           "none"
         ],
         # hugging face model ID.
         "model_id": "NumbersStation/nsql-2B"
       }
     ],
     # prompt style, required by chat models.
     # for more details, see: xinference/model/llm/tests/test_utils.py
     "prompt_style": None
   }

Register the custom model:

.. code-block:: python

   import json
   from xinference.client import Client

   # replace with real xinference endpoint
   endpoint = "http://localhost:9997"
   client = Client(endpoint)
   client.register_model(model_type="LLM", model=json.dumps(custom_model), persist=False)

Load the custom model:

.. code-block:: python

   uid = client.launch_model(model_name='nsql-2B')

Run the custom model:

.. code-block:: python

   text = """CREATE TABLE work_orders (
       ID NUMBER,
       CREATED_AT TEXT,
       COST FLOAT,
       INVOICE_AMOUNT FLOAT,
       IS_DUE BOOLEAN,
       IS_OPEN BOOLEAN,
       IS_OVERDUE BOOLEAN,
       COUNTRY_NAME TEXT,
   )

   -- Using valid SQLite, answer the following questions for the tables provided above.

   -- how many work orders are open?

   SELECT"""

   model = client.get_model(model_uid=uid)
   model.generate(prompt=text)

Result:

.. code-block:: json

   {
      "id":"aeb5c87a-352e-11ee-89ad-9af9f16816c5",
      "object":"text_completion",
      "created":1691418511,
      "model":"3b912fc4-352e-11ee-8e66-9af9f16816c5",
      "choices":[
         {
            "text":" COUNT(*) FROM work_orders WHERE IS_OPEN = '1';",
            "index":0,
            "logprobs":"None",
            "finish_reason":"stop"
         }
      ],
      "usage":{
         "prompt_tokens":117,
         "completion_tokens":17,
         "total_tokens":134
      }
   }

Custom models from URI
~~~~~~~~~~~~~~~~~~~~~~
Coming soon.