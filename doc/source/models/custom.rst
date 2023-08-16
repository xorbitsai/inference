.. _models_custom:

============================
Custom Models (Experimental)
============================

Custom models are currently an experimental feature and are expected to be officially released in
version v0.2.0.

Define a custom model
~~~~~~~~~~~~~~~~~~~~~

Define a custom model based on the following template:

.. code-block:: python

   custom_model = {
     "version": 1,
     # model name. must start with a letter or a
     # digit, and can only contain letters, digits,
     # underscores, or dashes.
     "model_name": "custom-llama-2",
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
         "model_size_in_billions": 7,
         # quantizations.
         "quantizations": [
           "4-bit",
           "8-bit",
           "none"
         ],
         # hugging face model ID.
         "model_id": "meta-llama/Llama-2-7b",
         # when model_uri is present, xinference will load the model from the given RUI.
         "model_uri": "file:///path/to/llama-2-7b"
       },
       {
         # model format.
         "model_format": "pytorch",
         "model_size_in_billions": 13,
         # quantizations.
         "quantizations": [
           "4-bit",
           "8-bit",
           "none"
         ],
         # hugging face model ID.
         "model_id": "meta-llama/Llama-2-13b"
       },
       {
         # model format.
         "model_format": "ggmlv3",
         # quantizations.
         "model_size_in_billions": 7,
         "quantizations": [
           "q4_0",
           "q8_0"
         ]
         # hugging face model ID.
         "model_id": "TheBloke/Llama-2-7B-GGML",
         # an f-string that takes a quantization.
         "model_file_name_template": "llama-2-7b.ggmlv3.{quantization}.bin"
       }
     ],
     # prompt style, required by chat models.
     # for more details, see: xinference/model/llm/tests/test_utils.py
     "prompt_style": None
   }

* model_name: A string defining the name of the model. The name must start with a letter or a digit and can only contain letters, digits, underscores, or dashes.
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


Register the Custom Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import json
   from xinference.client import Client

   # replace with real xinference endpoint
   endpoint = "http://localhost:9997"
   client = Client(endpoint)
   client.register_model(model_type="LLM", model=json.dumps(custom_model), persist=False)



Load the Custom Model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   uid = client.launch_model(model_name='custom-llama-2')

Run the Custom Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = client.get_model(model_uid=uid)
   model.generate("What is the largest animal in the world?")

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
