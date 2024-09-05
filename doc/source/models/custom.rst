.. _models_custom:

=============
Custom Models
=============
Xinference provides a flexible and comprehensive way to integrate, manage, and utilize custom models.


Directly launch an existing model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since ``v0.14.0``, you can directly launch an existing model by passing ``model_path`` to the launch interface without downloading it.
This way requires that the model's ``model_family`` is among the built-in supported models,
and eliminates the hassle of registering the model.

For example:

.. tabs::

  .. code-tab:: bash shell

    xinference launch --model_path <model_file_path> --model-engine <engine> -n qwen1.5-chat

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://127.0.0.1:9997/v1/models' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model_engine": "<engine>",
      "model_name": "qwen1.5-chat",
      "model_path": "<model_file_path>"
    }'

  .. code-tab:: python

    from xinference.client import RESTfulClient
    client = RESTfulClient("http://127.0.0.1:9997")
    model_uid = client.launch_model(
      model_engine="<inference_engine>",
      model_name="qwen1.5-chat",
      model_path="<model_file_path>"
    )
    print('Model uid: ' + model_uid)


The above example demonstrates how to directly launch a qwen1.5-chat model file without registering it.

For distributed scenarios, if your model file is on a specific worker,
you can directly launch it using the ``worker_ip`` and ``model_path`` parameters with the launch interface.

Define a custom LLM model
~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom LLM model based on the following template:

.. code-block:: json

   {
     "version": 1,
     "context_length": 2048,
     "model_name": "custom-llama-2-chat",
     "model_lang": [
       "en"
     ],
     "model_ability": [
       "chat"
     ],
     "model_family": "my-llama-2-chat",
     "model_specs": [
       {
         "model_format": "pytorch",
         "model_size_in_billions": 7,
         "quantizations": [
           "none"
         ],
         "model_uri": "file:///path/to/llama-2-chat"
       },
       {
         "model_format": "ggufv2",
         "model_size_in_billions": 7,
         "quantizations": [
           "q4_0",
           "q8_0"
         ],
         "model_file_name_template": "llama-2-chat-7b.{quantization}.gguf"
         "model_uri": "file:///path/to/gguf-file"
       }
     ],
     "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '<s>' + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content | trim + ' ' + '</s>' }}{% endif %}{% endfor %}",
     "stop_token_ids": [2],
     "stop": []
   }

* model_name: A string defining the name of the model. The name must start with a letter or a digit and can only contain letters, digits, underscores, or dashes.
* context_length: context_length: An optional integer that specifies the maximum context size the model was trained to accommodate, encompassing both the input and output lengths. If not defined, the default value is 2048 tokens (~1,500 words).
* model_lang: A list of strings representing the supported languages for the model. Example: ["en"], which means that the model supports English.
* model_ability: A list of strings defining the abilities of the model. It could include options like "embed", "generate", and "chat". In this case, the model has the ability to "generate".
* model_family: A required string representing the family of the model you want to register. This parameter must not conflict with any builtin model names.
* model_specs: An array of objects defining the specifications of the model. These include:
   * model_format: A string that defines the model format, like "pytorch" or "ggufv2".
   * model_size_in_billions: An integer defining the size of the model in billions of parameters.
   * quantizations: A list of strings defining the available quantizations for the model. For PyTorch models, it could be "4-bit", "8-bit", or "none". For ggufv2 models, the quantizations should correspond to values that work with the ``model_file_name_template``.
   * model_id: A string representing the model ID, possibly referring to an identifier used by Hugging Face. **If model_uri is missing, Xinference will try to download the model from the huggingface repository specified here.**.
   * model_uri: A string representing the URI where the model can be loaded from, such as "file:///path/to/llama-2-7b". **When the model format is ggufv2, model_uri must be the specific file path. When the model format is pytorch, model_uri must be the path to the directory containing the model files.** If model URI is absent, Xinference will try to download the model from Hugging Face with the model ID.
   * model_file_name_template: Required by gguf models. An f-string template used for defining the model file name based on the quantization. **Note that this field is just a template for the format of the ggufv2 model file, do not fill in the specific path of the model file.**
* chat_template: If ``model_ability`` includes ``chat`` , you must configure this option to generate the correct full prompt during chat. This is a Jinja template string. Usually, you can find it in the ``tokenizer_config.json`` file within the model directory.
* stop_token_ids: If ``model_ability`` includes ``chat`` , you can configure this option to control when the model stops during chat. This is a list of integers, and you can typically extract the corresponding values from the ``generation_config.json`` or ``tokenizer_config.json`` file in the model directory.
* stop: If ``model_ability`` includes ``chat`` , you can configure this option to control when the model stops during chat. This is a list of strings, and you can typically extract the corresponding values from the ``generation_config.json`` or ``tokenizer_config.json`` file in the model directory.

Define a custom embedding model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom embedding model based on the following template:

.. code-block:: json

    {
        "model_name": "custom-bge-base-en",
        "dimensions": 768,
        "max_tokens": 512,
        "language": ["en"],
        "model_id": "BAAI/bge-base-en",
        "model_uri": "file:///path/to/bge-base-en"
    }

* model_name: A string defining the name of the model. The name must start with a letter or a digit and can only contain letters, digits, underscores, or dashes.
* dimensions: A integer that specifies the embedding dimensions.
* max_tokens: A integer that represents the max sequence length that the embedding model supports.
* language: A list of strings representing the supported languages for the model. Example: ["en"], which means that the model supports English.
* model_id: A string representing the model ID, possibly referring to an identifier used by Hugging Face.
* model_uri: A string representing the URI where the model can be loaded from, such as "file:///path/to/your_model". If model URI is absent, Xinference will try to download the model from Hugging Face with the model ID.


Define a custom Rerank model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom rerank model based on the following template:

.. code-block:: json

    {
        "model_name": "custom-bge-reranker-v2-m3",
        "type": "normal",
        "language": ["en", "zh", "multilingual"],
        "model_id": "BAAI/bge-reranker-v2-m3",
        "model_uri": "file:///path/to/bge-reranker-v2-m3"
    }

* model_name: A string defining the name of the model. The name must start with a letter or a digit and can only contain letters, digits, underscores, or dashes.
* type: A string defining the type of the model, including ``normal``, ``LLM-based`` and ``LLM-based layerwise``.
* language: A list of strings representing the supported languages for the model. Example: ["en"], which means that the model supports English.
* model_id: A string representing the model ID, possibly referring to an identifier used by Hugging Face.
* model_uri: A string representing the URI where the model can be loaded from, such as "file:///path/to/your_model". If model URI is absent, Xinference will try to download the model from Hugging Face with the model ID.


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
   client.register_model(model_type="<model_type>", model=model, persist=False)

Or via CLI:

.. code-block:: bash

   xinference register --model-type <model_type> --file model.json --persist

Note that replace the ``<model_type>`` above with ``LLM``, ``embedding`` or ``rerank``. The same as below.


List the Built-in and Custom Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List built-in and custom models programmatically:

.. code-block:: python

   registrations = client.list_model_registrations(model_type="<model_type>")

Or via CLI:

.. code-block:: bash

   xinference registrations --model-type <model_type>

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

   model = client.unregister_model(model_type="<model_type>", model_name='custom-llama-2')

Or via CLI:

.. code-block:: bash

   xinference unregister --model-type <model_type> --model-name custom-llama-2
