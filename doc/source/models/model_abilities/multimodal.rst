.. _multimodal:

=====================
Multimodal
=====================

Learn how to process images and audio with LLMs.


Vision
============

With the ``vision`` ability you can have your model take in images and answer questions about them.
Within Xinference, this indicates that certain models are capable of processing image inputs when conducting
dialogues via the Chat API.


Supported models
----------------------

The ``vision`` ability is supported with the following models in Xinference:

* :ref:`qwen-vl-chat <models_llm_qwen-vl-chat>`
* :ref:`deepseek-vl-chat <models_llm_deepseek-vl-chat>`
* :ref:`yi-vl-chat <models_llm_yi-vl-chat>`
* :ref:`omnilmm <models_llm_omnilmm>`
* :ref:`internvl-chat <models_llm_internvl-chat>`
* :ref:`cogvlm2 <models_llm_cogvlm2>`
* :ref:`MiniCPM-Llama3-V 2.5 <models_llm_minicpm-llama3-v-2_5>`
* :ref:`GLM-4V <models_llm_glm-4v>`
* :ref:`MiniCPM-Llama3-V 2.6 <models_llm_minicpm-v-2.6>`
* :ref:`internvl2 <models_llm_internvl2>`
* :ref:`qwen2-vl-instruct <models_llm_qwen2-vl-instruct>`
* :ref:`llama-3.2-vision <models_llm_llama-3.2-vision>`
* :ref:`llama-3.2-vision-instruct <models_llm_llama-3.2-vision-instruct>`
* :ref:`glm-edge-v <models_llm_glm-edge-v>`


Quickstart
----------------------

Images are made available to the model in two main ways: by passing a link to the image or by passing the
base64 encoded image directly in the request.

Example using OpenAI Client
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import openai

    client = openai.Client(
        api_key="cannot be empty", 
        base_url=f"http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    response = client.chat.completions.create(
        model="<MODEL_UID>",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "http://i.epochtimes.com/assets/uploads/2020/07/shutterstock_675595789-600x400.jpg",
                        },
                    },
                ],
            }
        ],
    )
    print(response.choices[0])


Uploading base 64 encoded images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import openai
    import base64

    # Function to encode the image
    def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    # Path to your image
    image_path = "path_to_your_image.jpg"

    # Getting the base64 string
    b64_img = encode_image(image_path)

    client = openai.Client(
        api_key="cannot be empty", 
        base_url=f"http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    response = client.chat.completions.create(
        model="<MODEL_UID>",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}",
                        },
                    },
                ],
            }
        ],
    )
    print(response.choices[0])


You can find more examples of ``vision`` ability in the tutorial notebook:

.. grid:: 1

   .. grid-item-card:: Qwen VL Chat
      :link: https://github.com/xorbitsai/inference/blob/main/examples/chat_vl.ipynb
      
      Learn vision ability from a example using qwen-vl-chat

Audio
============

With the ``audio`` ability you can have your model take in audio and performing audio analysis or direct textual
responses with regard to speech instructions.
Within Xinference, this indicates that certain models are capable of processing audio inputs when conducting
dialogues via the Chat API.

Supported models
----------------------

The ``audio`` ability is supported with the following models in Xinference:

* :ref:`qwen2-audio-instruct <models_llm_qwen2-audio-instruct>`

Quickstart
----------------------

Audios are made available to the model in two main ways: by passing a link to the image or by passing the
audio url directly in the request.


Chat with audio
~~~~~~~~~~~~~~~

.. code-block:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model(<MODEL_UID>)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                },
                {"type": "text", "text": "What's that sound?"},
            ],
        },
        {"role": "assistant", "content": "It is the sound of glass shattering."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What can you do when you hear that?"},
            ],
        },
        {
            "role": "assistant",
            "content": "Stay alert and cautious, and check if anyone is hurt or if there is any damage to property.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac",
                },
                {"type": "text", "text": "What does the person say?"},
            ],
        },
    ]
    print(model.chat(messages))
