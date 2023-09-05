.. _models_custom:

=============
Custom Models
=============
Xinference provides a flexible and comprehensive way to integrate, manage, and utilize custom models.

Define a custom model
~~~~~~~~~~~~~~~~~~~~~

Define a custom model based on the following template:

.. code-block:: json

   {
     "version": 1,
     "context_length": 2048,
     "model_name": "custom-llama-2",
     "model_lang": [
       "en"
     ],
     "model_ability": [
       "generate"
     ],
     "model_specs": [
       {
         "model_format": "pytorch",
         "model_size_in_billions": 7,
         "quantizations": [
           "4-bit",
           "8-bit",
           "none"
         ],
         "model_id": "meta-llama/Llama-2-7b",
         "model_uri": "file:///path/to/llama-2-7b"
       },
       {
         "model_format": "ggmlv3",
         "model_size_in_billions": 7,
         "quantizations": [
           "q4_0",
           "q8_0"
         ],
         "model_id": "TheBloke/Llama-2-7B-GGML",
         "model_file_name_template": "llama-2-7b.ggmlv3.{quantization}.bin"
       }
     ]
   }

* model_name: A string defining the name of the model. The name must start with a letter or a digit and can only contain letters, digits, underscores, or dashes.
* context_length: context_length: An optional integer that specifies the maximum context size the model was trained to accommodate, encompassing both the input and output lengths. If not defined, the default value is 2048 tokens (~1,500 words).
* model_lang: A list of strings representing the supported languages for the model. Example: ["en"], which means that the model supports English.
* model_ability: A list of strings defining the abilities of the model. It could include options like "embed", "generate", and "chat". In this case, the model has the ability to "generate".
* model_specs: An array of objects defining the specifications of the model. These include:
   * model_format: A string that defines the model format, could be "pytorch" or "ggmlv3".
   * model_size_in_billions: An integer defining the size of the model in billions of parameters.
   * quantizations: A list of strings defining the available quantizations for the model. For PyTorch models, it could be "4-bit", "8-bit", or "none". For ggmlv3 models, the quantizations should correspond to values that work with the ``model_file_name_template``.
   * model_id: A string representing the model ID, possibly referring to an identifier used by Hugging Face.
   * model_uri: A string representing the URI where the model can be loaded from, such as "file:///path/to/llama-2-7b". If model URI is absent, Xinference will try to download the model from Hugging Face with the model ID.
   * model_file_name_template: Required by ggml models. An f-string template used for defining the model file name based on the quantization.
* prompt_style: An optional field that could be required by chat models to define the style of prompts. The given example has this set to None, but additional details could be found in a referenced file xinference/model/llm/tests/test_utils.py.


Register a Custom Model
~~~~~~~~~~~~~~~~~~~~~~~

Register a custom model programmatically:

.. code-block:: python

   import json
   from xinference.client import Client

   with open('model.json') as fd:
       model = fd.read()

   # replace with real xinference endpoint
   endpoint = 'http://localhost:9997'
   client = Client(endpoint)
   client.register_model(model_type="LLM", model=model, persist=False)

Or via CLI:

.. code-block:: bash

   xinference register --model-type LLM --file model.json --persist

List the Built-in and Custom Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List built-in and custom models programmatically:

.. code-block:: python

   registrations = client.list_model_registrations(model_type="LLM")

Or via CLI:

.. code-block:: bash

   xinference registrations --model-type LLM

Launch the Custom Model
~~~~~~~~~~~~~~~~~~~~~~~

Launch the custom model programmatically:

.. code-block:: python

   uid = client.launch_model(model_name='custom-llama-2', model_format='pytorch')

Or via CLI:

.. code-block:: bash

   xinference launch --model-name custom-llama-2 --model-format pytorch

Interact with the Custom Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Invoke the model programmatically:

.. code-block:: python

   model = client.get_model(model_uid=uid)
   model.generate('What is the largest animal in the world?')

Result:

.. code-block:: json

   {
      "id":"cmpl-a4a9d9fc-7703-4a44-82af-fce9e3c0e52a",
      "object":"text_completion",
      "created":1692024624,
      "model":"43e1f69a-3ab0-11ee-8f69-fa163e74fa2d",
      "choices":[
         {
            "text":"\nWhat does an octopus look like?\nHow many human hours has an octopus been watching you for?",
            "index":0,
            "logprobs":"None",
            "finish_reason":"stop"
         }
      ],
      "usage":{
         "prompt_tokens":10,
         "completion_tokens":23,
         "total_tokens":33
      }
   }

Or via CLI, replace ``${UID}`` with real model UID:

.. code-block:: bash

   xinference generate --model-uid ${UID}

Unregister the Custom Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unregister the custom model programmatically:

.. code-block:: python

   model = client.unregister_model(model_type='LLM', model_name='custom-llama-2')

Or via CLI:

.. code-block:: bash

   xinference unregister --model-type LLM --model-name custom-llama-2
