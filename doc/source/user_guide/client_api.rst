.. _user_guide_client_api:

==========
Client API
==========

Full Client API References: :ref:`reference_index`

In order to use Client, user need to start the xinference server by the following command line:

.. code-block::

    >>> xinference
    2023-10-17 16:32:21,700 xinference   24584 INFO     Xinference successfully started. Endpoint: http://127.0.0.1:9997
    2023-10-17 16:32:21,700 xinference.core.supervisor 24584 INFO     Worker 127.0.0.1:62590 has been added successfully
    2023-10-17 16:32:21,701 xinference.deploy.worker 24584 INFO     Xinference worker successfully started.

As the log shows above, the endpoint is `http://127.0.0.1:9997`. Users can use Client to connect 
to the xinference server through this endpoint.


Models are divided into these categories: LLM, embedding, image, ... More types of models will be added in the future.


LLM
~~~

List available built-in LLM models:

.. code-block::

    >>> xinference registrations -t LLM

    Type    Name                     Language      Ability                        Is-built-in
    ------  -----------------------  ------------  -----------------------------  -------------
    LLM     baichuan                 ['en', 'zh']  ['embed', 'generate']          True
    LLM     baichuan-2               ['en', 'zh']  ['embed', 'generate']          True
    LLM     baichuan-2-chat          ['en', 'zh']  ['embed', 'generate', 'chat']  True
    ...


Launch a LLM model and chat:


.. code-block::

    from xinference.client import Client

    client = Client("http://localhost:9997")
    # The chatglm2 model has the capabilities of "chat" and "embed".
    model_uid = client.launch_model(model_name="chatglm2", 
                                    model_format="ggmlv3",
                                    model_size_in_billions=6,
                                    quantization="q4_0")
    model = client.get_model(model_uid)

    chat_history = []
    prompt = "What is the largest animal?"
    # If the model has "generate" capability, then you can call
    # model.generate API.
    model.chat(
        prompt,
        chat_history,
        generate_config={"max_tokens": 1024}
    )


Embedding
~~~~~~~~~

List available built-in embedding models:

.. code-block::

    >>> xinference registrations -t embedding

    Type       Name                     Language      Dimensions  Is-built-in
    ---------  -----------------------  ----------  ------------  -------------
    embedding  bge-base-en              ['en']               768  True
    embedding  bge-base-en-v1.5         ['en']               768  True
    embedding  bge-base-zh              ['zh']               768  True
    ...


Launch an embedding model and embed a text:

.. code-block::

    from xinference.client import Client

    client = Client("http://localhost:9997")
    # The bge-small-en-v1.5 is an embedding model, so the `model_type` needs to be specified.
    model_uid = client.launch_model(model_name="bge-small-en-v1.5", model_type="embedding")
    model = client.get_model(model_uid)

    input_text = "what is the capital of China?"
    model.create_embedding(input_text)

Output

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

Image
~~~~~

List available built-in image models:

.. code-block::

    >>> xinference registrations -t image

    Type    Name                          Family            Is-built-in
    ------  ----------------------------  ----------------  -------------
    image   stable-diffusion-v1.5         stable_diffusion  True
    image   stable-diffusion-xl-base-1.0  stable_diffusion  True


Launch an image model and generate an image by prompt:

.. code-block::
    
    from xinference.client import Client

    client = Client("http://localhost:9997")
    # The stable-diffusion-v1.5 is an image model, so the `model_type` needs to be specified.
    # Additional kwargs can be passed to AutoPipelineForText2Image.from_pretrained here.
    model_uid = client.launch_model(model_name="stable-diffusion-v1.5", model_type="image")
    model = client.get_model(model_uid)

    input_text = "an apple"
    model.text_to_image(input_text)

Output

.. code-block::

    {'created': 1697536913,
     'data': [{'url': '/home/admin/.xinference/image/605d2f545ac74142b8031455af31ee33.jpg',
     'b64_json': None}]}
