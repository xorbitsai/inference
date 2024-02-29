.. _index:

======================
Welcome to Xinference!
======================

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   models/index
   user_guide/index
   examples/index
   reference/index


Xorbits Inference (Xinference) is an open-source platform to streamline the operation and integration
of a wide array of AI models. With Xinference, you're empowered to run inference using any open-source LLMs,
embedding models, and multimodal models either in the cloud or on your own premises, and create robust
AI-driven applications.   

Developing Real-world AI Applications with Xinference
-----------------------------------------------------

.. tabs::

  .. code-tab:: python LLM

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    # Chat to LLM
    model.chat(
       prompt="What is the largest animal?",
       system_prompt="You are a helpful assistant",
       generate_config={"max_tokens": 1024}
    )
    
    # Chat to VL model
    model.chat(
       chat_history=[
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
      generate_config={"max_tokens": 1024}
    )    

  .. code-tab:: python Embedding

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    model.create_embedding("What is the capital of China?")

  .. code-tab:: python Image

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    model.text_to_image("An astronaut walking on the mars")

  .. code-tab:: python Audio

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    with open("speech.mp3", "rb") as audio_file:
        model.transcriptions(audio_file.read())

  .. code-tab:: python Rerank

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    query = "A man is eating pasta."
    corpus = [
      "A man is eating food.",
      "A man is eating a piece of bread.",
      "The girl is carrying a baby.",
      "A man is riding a horse.",
      "A woman is playing violin."
    ]
    print(model.rerank(corpus, query))


Getting Started
---------------

.. grid:: 2

    .. grid-item-card::  Install Xinference
      :link: installation
      :link-type: ref

      Install Xinference on Linux, Windows, and macOS.

    .. grid-item-card::  Try it out!
      :link: using_xinference
      :link-type: ref

      Start by running Xinference on a local machine.


.. grid:: 2

   .. grid-item-card:: Explore models
      :link: models_builtin_index
      :link-type: ref
      
      Explore a wide range of models supported by Xinference.

   .. grid-item-card:: Register your own model
      :link: models_custom
      :link-type: ref
      
      Register model weights and turn it into an API.



Explore the API
---------------

.. grid:: 2

    .. grid-item-card::  Chat & Generate
      :link: chat
      :link-type: ref

      Learn how to chat with LLMs in Xinference.

    .. grid-item-card::  Tools
      :link: tools
      :link-type: ref

      Learn how to connect LLM with external tools.


.. grid:: 2

    .. grid-item-card::  Embeddings
      :link: embed
      :link-type: ref

      Learn how to create text embeddings in Xinference.

    .. grid-item-card::  Rerank
      :link: rerank
      :link-type: ref

      Learn how to use rerank models in Xinference.


.. grid:: 2

    .. grid-item-card::  Images
      :link: image
      :link-type: ref

      Learn how to generate images with Xinference.

    .. grid-item-card::  Vision
      :link: vision
      :link-type: ref

      Learn how to process image with LLMs.


.. grid:: 2

    .. grid-item-card::  Audio
      :link: audio
      :link-type: ref

      Learn how to turn audio into text or text into audio with Xinference.


Getting Involved
----------------

.. grid:: 
   :gutter: 1

   .. grid-item::
      
      .. div:: sd-font-weight-normal sd-fs-5
         
         Get Latest News

      .. grid:: 1
         :gutter: 3

         .. grid-item-card::  
            :link: https://twitter.com/Xorbitsio

            :fab:`twitter` Follow us on Twitter

         .. grid-item-card::  
            :link: https://zhihu.com/org/xorbits

            :fab:`zhihu` Read our blogs


   .. grid-item::      

      .. div:: sd-font-weight-normal sd-fs-5

         Get Support

      .. grid:: 1
         :gutter: 3

         .. grid-item-card:: 
            :link: https://xorbits.cn/assets/images/wechat_work_qr.png
            
            :fab:`weixin` Find community on WeChat

         .. grid-item-card:: 
            :link: https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg
            
            :fab:`slack` Find community on Slack

         .. grid-item-card::  
            :link: https://github.com/xorbitsai/inference/issues/new/choose

            :fab:`github` Open an issue


   .. grid-item::      

      .. div:: sd-fs-5

         Contribute to Xinference

      .. grid:: 1
         :gutter: 3

         .. grid-item-card::  
            :link: https://github.com/xorbitsai/inference/pulls

            :fab:`github` Create a pull request