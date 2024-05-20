.. _user_guide_client_api:

==========
Client API
==========

Complete Client API Reference: :ref:`reference_index`

To utilize the Client API, initiate the xinference server using the command below:

.. code-block::

    >>> xinference
    2023-10-17 16:32:21,700 xinference   24584 INFO     Xinference successfully started. Endpoint: http://127.0.0.1:9997
    2023-10-17 16:32:21,700 xinference.core.supervisor 24584 INFO     Worker 127.0.0.1:62590 has been added successfully
    2023-10-17 16:32:21,701 xinference.deploy.worker 24584 INFO     Xinference worker successfully started.

Based on the log above, the endpoint is `http://127.0.0.1:9997`. Users can connect to the xinference server through this endpoint using the Client.

Models are categorized into LLM, embedding, image, etc. We plan to introduce more model types in the future.

LLM
~~~

To list the available built-in LLM models:

.. code-block::

    >>> xinference registrations -t LLM

    Type    Name                     Language      Ability                        Is-built-in
    ------  -----------------------  ------------  -----------------------------  -------------
    LLM     baichuan                 ['en', 'zh']  ['embed', 'generate']          True
    LLM     baichuan-2               ['en', 'zh']  ['embed', 'generate']          True
    LLM     baichuan-2-chat          ['en', 'zh']  ['embed', 'generate', 'chat']  True
    ...

To initialize an LLM and chat:

Xinference Client
=================

.. code-block::

    from xinference.client import Client

    client = Client("http://localhost:9997")
    # The chatglm2 model has the capabilities of "chat" and "embed".
    model_uid = client.launch_model(model_name="chatglm2",
                                    model_engine="llama.cpp",
                                    model_format="ggmlv3",
                                    model_size_in_billions=6,
                                    quantization="q4_0")
    model = client.get_model(model_uid)

    chat_history = []
    prompt = "What is the largest animal?"
    # If the model has "generate" capability, then you can call the
    # model.generate API.
    model.chat(
        prompt,
        chat_history=chat_history,
        generate_config={"max_tokens": 1024}
    )

OpenAI Client
=============

Openai client request with the same function as before, excluding launch model. 
More details refer to: https://platform.openai.com/docs/api-reference/chat?lang=python

.. code-block::

    import openai

    # Assume that the model is already launched.
    # The api_key can't be empty, any string is OK.
    client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
    client.chat.completions.create(
        model=model_uid,
        messages=[
            {
                "content": "What is the largest animal?",
                "role": "user",
            }
        ],
        max_tokens=1024
    )

OpenAI Client Tool Calls
========================

.. code-block::

    import openai

    tools = [
        {
            "type": "function",
            "function": {
                "name": "uber_ride",
                "description": "Find suitable ride for customers given the location, "
                "type of ride, and the amount of time the customer is "
                "willing to wait as parameters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "loc": {
                            "type": "int",
                            "description": "Location of the starting place of the Uber ride",
                        },
                        "type": {
                            "type": "string",
                            "enum": ["plus", "comfort", "black"],
                            "description": "Types of Uber ride user is ordering",
                        },
                        "time": {
                            "type": "int",
                            "description": "The amount of time in minutes the customer is willing to wait",
                        },
                    },
                },
            },
        }
    ]

    # Assume that the model is already launched.
    # The api_key can't be empty, any string is OK.
    client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
    client.chat.completions.create(
        model="chatglm3",
        messages=[{"role": "user", "content": "Call me an Uber ride type 'Plus' in Berkeley at zipcode 94704 in 10 minutes"}],
        tools=tools,
    )

Output:

.. code-block::

    ChatCompletion(id='chatcmpl-ad2f383f-31c7-47d9-87b7-3abe928e629c', choices=[Choice(finish_reason='tool_calls', index=0, message=ChatCompletionMessage(content="```python\ntool_call(loc=94704, type='plus', time=10)\n```", role='assistant', function_call=None, tool_calls=[ChatCompletionMessageToolCall(id='call_ad2f383f-31c7-47d9-87b7-3abe928e629c', function=Function(arguments='{"loc": 94704, "type": "plus", "time": 10}', name='uber_ride'), type='function')]))], created=1704687803, model='chatglm3', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=-1, prompt_tokens=-1, total_tokens=-1))



Embedding
~~~~~~~~~

To list the available built-in embedding models:

.. code-block::

    >>> xinference registrations -t embedding

    Type       Name                     Language      Dimensions  Is-built-in
    ---------  -----------------------  ----------  ------------  -------------
    embedding  bge-base-en              ['en']               768  True
    embedding  bge-base-en-v1.5         ['en']               768  True
    embedding  bge-base-zh              ['zh']               768  True
    ...

To launch an embedding model and embed text:

Xinference Client
=================

.. code-block::

    from xinference.client import Client

    client = Client("http://localhost:9997")
    # The bge-small-en-v1.5 is an embedding model, so the `model_type` needs to be specified.
    model_uid = client.launch_model(model_name="bge-small-en-v1.5", model_type="embedding")
    model = client.get_model(model_uid)

    input_text = "What is the capital of China?"
    model.create_embedding(input_text)

Output:

.. code-block::

    {'object': 'list',
     'model': 'da2a511c-6ccc-11ee-ad07-22c9969c1611-1-0',
     'data': [{'index': 0,
     'object': 'embedding',
     'embedding': [-0.014207549393177032,
        -0.01832585781812668,
        0.010556723922491074,
        ...
        -0.021243810653686523,
        -0.03009396605193615,
        0.05420297756791115]}],
     'usage': {'prompt_tokens': 37, 'total_tokens': 37}}

OpenAI Client
=============

Openai client request with the same function as before, excluding launch model. 
More details refer to: https://platform.openai.com/docs/api-reference/embeddings?lang=python

.. code-block::

    import openai

    # Assume that the model is already launched.
    # The api_key can't be empty, any string is OK.
    client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
    client.embeddings.create(model=model_uid, input=["What is the capital of China?"])

Output:

.. code-block::

    CreateEmbeddingResponse(data=[Embedding(embedding=[-0.014207549393177032, -0.01832585781812668, 0.010556723922491074, ..., -0.021243810653686523, -0.03009396605193615, 0.05420297756791115], index=0, object='embedding')], model='bge-small-en-v1.5-1-0', object='list', usage=Usage(prompt_tokens=37, total_tokens=37))

Image
~~~~~

To list the available built-in image models:

.. code-block::

    >>> xinference registrations -t image

    Type    Name                          Family            Is-built-in
    ------  ----------------------------  ----------------  -------------
    image   sd-turbo                      stable_diffusion  True
    image   sdxl-turbo                    stable_diffusion  True
    image   stable-diffusion-v1.5         stable_diffusion  True
    image   stable-diffusion-xl-base-1.0  stable_diffusion  True

To initiate an image model and generate an image using a text prompt:

Xinference Client
=================

.. code-block::

    from xinference.client import Client

    client = Client("http://localhost:9997")
    # The stable-diffusion-v1.5 is an image model, so the `model_type` needs to be specified.
    # Additional kwargs can be passed to AutoPipelineForText2Image.from_pretrained here.
    model_uid = client.launch_model(model_name="stable-diffusion-v1.5", model_type="image")
    model = client.get_model(model_uid)

    input_text = "an apple"
    model.text_to_image(input_text)

Output:

.. code-block::

    {'created': 1697536913,
     'data': [{'url': '/home/admin/.xinference/image/605d2f545ac74142b8031455af31ee33.jpg',
     'b64_json': None}]}

OpenAI Client
=============

Openai client request with the same function as before, excluding launch model. 
More details refer to: https://platform.openai.com/docs/api-reference/images/create?lang=python

.. code-block::

    import openai

    # Assume that the model is already launched.
    # The api_key can't be empty, any string is OK.
    client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
    client.images.generate(model=model_uid, prompt="an apple")


Output:

.. code-block::

    ImagesResponse(created=1704445354, data=[Image(b64_json=None, revised_prompt=None, url='/home/admin/.xinference/image/605d2f545ac74142b8031455af31ee33.jpg')])


Audio
~~~~~

To list the available built-in image models:

.. code-block::

    >>> xinference registrations -t audio

    Type    Name               Family    Multilingual    Is-built-in
    ------  -----------------  --------  --------------  -------------
    audio   whisper-base       whisper   True            True
    audio   whisper-base.en    whisper   False           True
    audio   whisper-large-v3   whisper   True            True
    audio   whisper-medium     whisper   True            True
    audio   whisper-medium.en  whisper   False           True
    audio   whisper-tiny       whisper   True            True
    audio   whisper-tiny.en    whisper   False           True


To initiate an audio model and get text from an audio:

Xinference Client
=================

.. code-block::

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model_uid = client.launch_model(model_name="whisper-large-v3", model_type="audio")
    model = client.get_model(model_uid)

    input_text = "an apple"
    with open("audio.mp3", "rb") as audio_file:
        model.transcriptions(audio_file.read())

Output:

.. code-block::

    {
      "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."
    }


OpenAI Client
=============

Openai client request with the same function as before.
More details refer to: https://platform.openai.com/docs/api-reference/audio/createTranscription

.. code-block::

    import openai

    # Assume that the model is already launched.
    # The api_key can't be empty, any string is OK.
    client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
    with open("audio.mp3", "rb") as audio_file:
        completion = client.audio.transcriptions.create(model=model_uid, file=audio_file)

Output:

.. code-block::

    Translation(text=' This list lists the airlines in Hong Kong.')


Rerank
~~~~~~
To launch a rerank model and compute the similarity scores:

.. code-block::

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model_uid = client.launch_model(model_name="bge-reranker-base", model_type="rerank")
    model = client.get_model(model_uid)

    query = "A man is eating pasta."
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin."
    ]
    print(model.rerank(corpus, query))

Output:

.. code-block::

    {'id': '480dca92-8910-11ee-b76a-c2c8e4cad3f5', 'results': [{'index': 0, 'relevance_score': 0.9999247789382935,
     'document': 'A man is eating food.'}, {'index': 1, 'relevance_score': 0.2564932405948639,
     'document': 'A man is eating a piece of bread.'}, {'index': 3, 'relevance_score': 3.955026841140352e-05,
     'document': 'A man is riding a horse.'}, {'index': 2, 'relevance_score': 3.742107219295576e-05,
     'document': 'The girl is carrying a baby.'}, {'index': 4, 'relevance_score': 3.739788007806055e-05,
     'document': 'A woman is playing violin.'}]}
