.. _vision:

=====================
Vision
=====================

Learn how to process images with LLMs.


Introduction
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


Quickstart
====================

Images are made available to the model in two main ways: by passing a link to the image or by passing the
base64 encoded image directly in the request.

Example using OpenAI Client
-------------------------------

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
------------------------------------

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


