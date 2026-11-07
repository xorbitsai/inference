.. _user_guide_client_api:

==========
Client API
==========

Complete Client API Reference: :ref:`reference_index`

Loaded context length
=====================

``client.get_model_runtime_info(model_uid)`` and its asynchronous equivalent call the Xinference-specific ``GET /v1/models/{model_uid}/runtime`` endpoint. ``runtime_context_length`` is the loaded adapter's total input-plus-output limit in tokens, or ``null`` if unknown. ``replicas`` lists each ``replica_uid`` and its limit. The model-wide value is the minimum only when every active replica reports a value. Queries follow the serving generation after reload; the model details panel fetches this information on entry and after a reload.

The registered ``context_length`` and OpenAI/Anthropic model, generation, and streaming response formats remain unchanged. vLLM, Transformers, MLX, and TRT-LLM report their adapter limit; SGLang queries its runtime. llama.cpp reads its resolved per-slot ``n_ctx`` from the loaded server's ``/props`` endpoint, rather than dividing the launch-time total capacity. Older runtimes without a query interface report ``null``. The limit does not guarantee available memory.

Replica management
==================

The synchronous and asynchronous clients accept ``replica_config`` in
``launch_model`` and expose ``add_model_replica`` and
``terminate_model_replica`` for running models. See :ref:`launch` for the
request structure, placement rules, and examples.

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
    model_uid = client.launch_model(model_name="glm4-chat",
                                    model_engine="llama.cpp",
                                    model_format="ggufv2",
                                    model_size_in_billions=9,
                                    quantization="Q4_K")
    model = client.get_model(model_uid)

    messages = [{"role": "user", "content": "What is the largest animal?"}]
    # If the model has "generate" capability, then you can call the
    # model.generate API.
    model.chat(
        messages,
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

Agent cache lifecycle
=====================

Supported Xavier adapters for vLLM hybrid models, SGLang, TRT-LLM and MLX advertise ``supports_agent_cache_lifecycle: true``. Attach ``cache_session={"id": "run-1", "revision": 0}`` to chat requests through the OpenAI client's ``extra_body`` or the Xinference client's ``generate_config``. Session IDs are scoped to the request credential. Cache isolation still uses ``cache_salt``.

A single replica can use ``enable_xavier=True`` and ``xavier_gpu_cache_bytes=0`` for lifecycle-aware resident prefixes. This mode requires vLLM >= 0.32 and a recurrent-hybrid model combining attention with recurrent state. Non-hybrid single-replica launches are rejected.

Send ``id``, an integer ``revision``, and ``phase`` (``waiting``, ``active`` or ``finished``) to ``POST /v1/models/{model_uid}/cache_sessions``. Chat marks the session active. Older revisions and phase changes at the same revision are rejected. Increase the revision after chat before sending ``waiting``. Repeating the same revision and phase refreshes the 300-second TTL. Xagent sends phase changes automatically; waits longer than five minutes require explicit refreshes.

Waiting prefixes receive a soft retention preference: vLLM reorders free hybrid GPU blocks; all four adapters refresh shared CPU prefixes, and LMCache can prefetch a bounded prefix from disk. Finished or expired sessions lose this preference without deleting shared pages. Blocks remain allocatable; active requests and transfer leases keep ownership. Storage failures fall back to prefill. P/D lifecycle hints are not supported.

The registry holds at most 4096 sessions per model and 1024 per credential scope. Requests without credentials share one anonymous scope. Lifecycle IDs are independent of ``cache_salt``; use distinct session IDs for independent workflows, including workflows with different salts.

Cache hit token counts
======================

With vLLM, SGLang, and MLX, completion and chat completion responses expose the engine's reused prompt-token count in ``usage.prompt_tokens_details.cached_tokens`` when available. This count includes native prefix-cache reuse and Xavier cache reuse reported by the engine; it is part of ``prompt_tokens``, not an additional token count.

With P/D disaggregation, this count includes KV transferred from the prefill engine, even when that KV was computed for the current request; it does not separately measure historical prefix-cache hits on the prefill engine.

For streaming requests, set ``stream_options={"include_usage": True}`` to receive the final usage chunk with ``choices=[]``. The Responses API exposes the same count as ``usage.input_tokens_details.cached_tokens``. If the engine does not report cache hits, completion and chat completion responses omit ``prompt_tokens_details``; the Responses API defaults the count to zero.

Chat replica affinity
=====================

For a model with ordinary chat replicas, send the same optional ``routing_key`` for each request in a session. Xinference prefers the last replica selected for that model, credential and key. If the replica is unavailable or at its configured request limit, Xinference tries another replica. Requests without a key use round-robin routing. The key must be a non-blank string of at most 128 characters and does not control cache isolation.

Set ``routing_key`` in ``extra_body`` with the OpenAI client or in ``generate_config`` with the Xinference client. The response from ``GET /v1/models/{model_uid}`` includes ``supports_chat_session_affinity: true`` for models that support this feature.

Xinference retains at most 4096 routing keys per model in a least-recently-used table shared across credentials. High key churn from one credential can evict another credential's affinity; this affects replica preference only.

With vLLM, affinity uses a positive integer ``max_num_seqs`` multiplied by the number of data-parallel ranks served by the replica as an advisory capacity. External load balancing serves one rank; hybrid load balancing counts local ranks. Invalid capacity values disable this engine-specific probe. Below this capacity, requests keep affinity and can batch. If all replicas are busy, Xinference keeps the preferred replica and its normal queueing and admission behavior. This probe counts requests, so requests with ``n > 1`` may consume more sequence slots than counted. Moving to another replica may incur prefix recomputation or KV transfer; the policy does not estimate these costs or reserve capacity.

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

.. _openai_responses_client:

OpenAI Client Responses API
===========================

    Responses API's access address is: /v1/responses

The Responses API is stateless in Xinference. Responses are not stored, so
``previous_response_id`` and ``conversation`` are rejected: send the whole
conversation in ``input`` on every request. Text and image input, function and
custom tools, reasoning, structured output and streaming are supported. Hosted
tools such as ``web_search`` are ignored.

.. code-block::

    import openai

    client = openai.Client(api_key="not empty", base_url="http://localhost:9997/v1")
    response = client.responses.create(
        model="qwen3",
        instructions="You are a helpful assistant.",
        input="What is the largest animal?",
    )
    print(response.output_text)

Codex CLI calls custom providers through this API. Add Xinference as a provider
in ``~/.codex/config.toml``:

.. code-block:: toml

    model = "qwen3"
    model_provider = "xinference"

    [model_providers.xinference]
    name = "Xinference"
    base_url = "http://localhost:9997/v1"
    wire_api = "responses"

.. _anthropic_client:

Anthropic Client
================

See :ref:`anthropic_api` for Claude Code setup, Python SDK examples, and
supported features and limitations.

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

To list the available built-in audio models:

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
    audio   speech_campplus_sv_zh-cn_16k-common                    campplus  False  True
    audio   speech_campplus_sv_zh_en_16k-common_advanced           campplus  True   True


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


Speaker Embeddings
==================

Speaker-embedding audio models extract a fixed-length representation of speaker
identity. Xinference provides two built-in CAMPPlus models:
``speech_campplus_sv_zh-cn_16k-common`` and
``speech_campplus_sv_zh_en_16k-common_advanced``. Both return a 192-dimensional
vector through the ``speaker_embedding`` ability.

Launch a model and call it with the Xinference client:

.. code-block:: python

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model_uid = client.launch_model(
        model_name="speech_campplus_sv_zh-cn_16k-common",
        model_type="audio",
    )
    model = client.get_model(model_uid)

    with open("speaker.wav", "rb") as audio_file:
        result = model.create_embedding(audio_file.read())

    print(result["dimensions"])
    embedding = result["embedding"]

The equivalent HTTP request uploads the model UID and audio file as multipart
form fields:

.. code-block:: bash

    curl -X POST 'http://localhost:9997/v1/audio/embeddings' \
      -H 'accept: application/json' \
      -F 'model=<MODEL_UID>' \
      -F 'file=@speaker.wav'

The response contains one vector rather than the list-shaped response returned
by the text Embeddings API:

.. code-block:: json

    {
      "object": "embedding",
      "model": "<MODEL_UID>",
      "dimensions": 192,
      "embedding": [0.0123, -0.0456, 0.0789]
    }

Compare vectors with cosine similarity for speaker verification or speaker
identification. See :ref:`audio` for Web UI usage, input processing details,
and the complete Speaker Embedding API example.


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


Request priority
================

All LLM engines (vLLM, SGLang, Transformers, MLX, MLX-VLM, llama.cpp, and TensorRT-LLM) support an integer ``priority`` from ``-10000`` to ``10000`` on ``/v1/chat/completions`` and ``/v1/completions``, including streaming requests. The default is ``0``; lower values have higher priority. Priority applies within a model replica; it does not select a replica or bypass request limits.

vLLM and SGLang use native priority scheduling. Enable it at model launch with ``scheduling_policy="priority"`` for vLLM (0.6.3 or later), or ``enable_priority_scheduling=True`` for a SGLang version that supports priority scheduling. Xinference converts SGLang priorities to its configured ordering, so lower values always mean higher priority in the Xinference API, regardless of ``schedule_low_priority_values_first``.

TRT-LLM versions exposing a native ``priority`` parameter use native scheduling for ordinary and P/D replicas. Xinference defaults ``scheduler_config.waiting_queue_policy`` to ``"priority"`` and converts the integer scale linearly: ``-10000`` maps to ``1.0``, ``0`` to ``0.5``, and ``10000`` to ``0.0``. Older versions, including 1.2.1, reject an explicitly supplied ``priority`` with HTTP 400 for non-streaming requests, even when it is ``0``. Omit the field to keep using those versions; no shared-queue fallback is applied to TRT-LLM.

Other engines use a per-replica priority queue before engine submission. Active requests are bounded by the engine's configured batch or parallel capacity; serial paths run one request at a time. Waiting requests run in priority order, with FIFO for equal priorities. This queue does not preempt running requests. Non-batching Transformers models now run one request at a time even when priority is omitted. Waiting requests count toward ``request_limits``; a full limit also rejects later requests with higher priority.

Xinference now checks client-supplied ``request_id`` values for all engines, including requests without ``priority``. IDs must be unique among active requests on a replica. After an abort, the ID is blocked for 30 seconds by default; use a fresh ID for retries, or configure ``block_duration`` in the abort request.

With the Xinference Python client, pass ``generate_config={"priority": -10}`` to ``model.chat()`` or ``model.generate()``. With the OpenAI Python client, use ``extra_body``:

.. code-block:: python

    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:9997/v1", api_key="your-api-key")
    response = client.chat.completions.create(
        model="your-model-uid",
        messages=[{"role": "user", "content": "Hello"}],
        extra_body={"priority": -10},
    )

Invalid priority types or values outside ``[-10000, 10000]`` return HTTP 422; strings, booleans, and fractional numbers are not converted to integers. Nonzero priorities require native priority scheduling to be enabled for vLLM, SGLang, and TRT-LLM versions with a native priority API; otherwise non-streaming requests fail with HTTP 400. For streaming requests, engine configuration and version checks happen after HTTP 200 is sent, so these errors appear in the SSE stream.


Chat completion errors
======================

For ``/v1/chat/completions``, unexpected worker failures return a generic error message. Server logs and error events retain the original diagnostics.

Before streaming starts, system-message ordering errors on strict templates and detected vLLM multimodal per-prompt limit violations return HTTP 400. If live limit lookup is unavailable, validation is deferred to the worker.

For non-streaming requests, recognized context-length violations and known media or cache-salt validation errors return HTTP 400; rate-limit errors return HTTP 429. Messages describe the failure without exposing worker details. Unrecognized worker failures return HTTP 500.

After a streaming response starts, its HTTP status remains 200. Failures are sent in an SSE ``error`` payload. Recognized context-length and rate-limit failures retain distinct safe messages; unexpected failures use the generic message.
