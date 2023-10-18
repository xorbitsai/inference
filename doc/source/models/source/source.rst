.. _models_download:

========
Download Source
========

Xinference supports downloading various models from different sources.

HuggingFace
^^^^^^^^^^^^^^
Xinference directly downloads the required models from the official `Hugging Face model repository <https://huggingface.co/models>`_ by default.

ModelScope
^^^^^^^^^^^^^^
Users can choose to download models from the `ModelScope model repository <https://modelscope.cn/models>`_.

Xinference supports downloading the following models from ModelScope:

* LLM Models
    * llama-2-chat
    * tiny-llama
    * baichuan-2-chat
    * baichuan-2
    * chatglm2
    * chatglm2-32k
    * internlm-7b
    * internlm-chat-7b
    * internlm-20b
    * internlm-chat-20b
    * wizardcoder-python-v1.0

* Embedding Models
    * bge-large-en
    * bge-base-en
    * gte-large
    * gte-base
    * e5-large-v2
    * bge-large-zh
    * bge-large-zh-noinstruct
    * bge-base-zh
    * multilingual-e5-large
    * bge-small-zh
    * bge-small-zh-v1.5
    * bge-base-zh-v1.5
    * bge-large-zh-v1.5
    * bge-small-en-v1.5
    * bge-base-en-v1.5
    * bge-large-en-v1.5


One of the following settings will make Xinference download models from ModelScope:

* The operating system's language is set to Simplified Chinese (zh_CN).
* Set the environment variable ``XINFERENCE_MODEL_SRC=modelscope``.
