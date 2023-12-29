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

  .. code-tab:: python Chat

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    model.chat(
       prompt = "What is the largest animal?",
       system_prompt = "You are a helpful assistant",
       chat_history=[],
       generate_config={"max_tokens": 1024}
    )

  .. code-tab:: python Embeddings

    from xinference.client import Client

    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    input_text = "What is the capital of China?"
    model.create_embedding("What is the capital of China?")

  .. code-tab:: python Images

    from xinference.client import Client
    client = Client("http://localhost:9997")
    model = client.get_model("MODEL_UID")

    model.text_to_image("An astronaut walking on the mars")


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
      :link: models_builtin_index
      :link-type: ref

      Install Xinference on Linux, Windows, and macOS.

    .. grid-item-card::  Try it out!
      :link: models_builtin_index
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
            :link: https://xorbits.cn/assets/images/wechat_pr.png
            
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