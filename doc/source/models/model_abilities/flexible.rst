.. _flexible:

====================================
Traditional ML models (Experimental)
====================================

Learn how to inference traditional machine learning models with Xinference.
These flexibly extensible models are referred to as **Flexible Models** within Xinference.

.. versionadded:: v1.7.1
  This ability is public since v1.7.1, now the API is not stable and may change during evolving.


Introduction
==================

Xinference provides flexible extensibility for performing inference with traditional machine learning models.
It includes built-in support for loading and running the following types of models:

- Hugging Face Pipelines for tasks such as classification using models hosted on Hugging Face.
- ModelScope Pipelines for tasks such as classification using models from ModelScope.
- YOLO for image detection and related computer vision tasks.

A wide range of traditional machine learning models can be used with Xinference.
For each of the categories above, we will walk through a representative example to
demonstrate how to perform inference step by step on the Xinference platform.

HuggingFace Pipeline Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we use `FacebookAI/roberta-large-mnli <https://huggingface.co/FacebookAI/roberta-large-mnli>`_ as an example.
This is a zero-shot classification model.
For other types of models, simply specify the corresponding task (which is also a parameter of the Pipeline)
when registering the model.

Download the model to the following path::

    /path/to/roberta-large-mnli

Next, we demonstrate how to register this flexible model in the Xinference Web UI.
For the following examples, unless we have to, we will skip the UI steps and focus on the core logic.

.. raw:: html

    <img class="align-center" alt="actor" src="../../_static/hf-pipeline.png" style="background-color: transparent", width="77%">

The corresponding custom model JSON file is as follows:

.. code-block:: json

    {
        "model_name": "roberta-large-mnli",
        "model_id": null,
        "model_revision": null,
        "model_hub": "huggingface",
        "model_description": "roberta-large-mnli is the RoBERTa large model fine-tuned on the Multi-Genre Natural Language Inference (MNLI) corpus. The model is a pretrained model on English language text using a masked language modeling (MLM) objective.",
        "model_uri": "/path/to/roberta-large-mnli",
        "launcher": "xinference.model.flexible.launchers.transformers",
        "launcher_args": "{\"task\": \"zero-shot-classification\"}",
        "virtualenv": {
            "packages": [],
            "inherit_pip_config": true,
            "index_url": null,
            "extra_index_url": null,
            "find_links": null,
            "trusted_host": null,
            "no_build_isolation": null
        },
        "is_builtin": false
    }

Refer to the section :ref:`register_custom_model` for instructions on registering the model using either code or the command line.

Next, load the model by selecting **Launch Model** / **Custom Model** / **Flexible Model** in the Web UI.
The loading procedure is the same as for other model types.

When using the command line, remember to specify the option `--model-type flexible`.

After the model is successfully loaded, we can perform inference using the following method.

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/flexible/infers' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "roberta-large-mnli",
        "args": [
          "one day I will see the world",
          ["travel", "cooking", "dancing"]
        ]
      }'

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("roberta-large-mnli")

    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    model.infer(sequence_to_classify, candidate_labels)


  .. code-tab:: json output

    {"sequence":"one day I will see the world","labels":["travel","cooking","dancing"],"scores":[0.9799638986587524,0.010605016723275185,0.009431036189198494]}

ModelScope Pipeline Model
~~~~~~~~~~~~~~~~~~~~~~~~~

ModelScope Pipeline models are very similar to Huggingface ones.
The only difference lies in the launcher used.

We take a zero-shot classification model from ModelScope as an example.
The model is `iic/nlp_structbert_zero-shot-classification_chinese-base <https://modelscope.cn/models/iic/nlp_structbert_zero-shot-classification_chinese-base>`_.

Here, we make use of Xinference's model virtual environment feature.
This is because the model used in this example requires `transformers==4.50.3` to run properly.
To isolate the environment, we use a :ref:`virtual env <model_virtual_env>` when registering the model.

When specifying custom packages during registration, the syntax is the same as for regular packages, with a few special cases.
Since the virtual environment is still based on the site packages of the Python runtime where Xinference is running, we need to explicitly include `#system_numpy#`.
Packages wrapped in `#system_xx#` ensure consistency with the base environment during virtual environment creation; otherwise, it may easily result in runtime errors.

Registering via Web UI:

.. raw:: html

    <img class="align-center" alt="actor" src="../../_static/modelscope-pipeline.png" style="background-color: transparent", width="77%">

Corresponding json file:

.. code-block:: json

    {
        "model_name": "nlp_structbert_zero-shot-classification_chinese-base",
        "model_id": null,
        "model_revision": null,
        "model_hub": "huggingface",
        "model_description": "This is a model description.",
        "model_uri": "/Users/xuyeqin/Downloads/models/nlp_structbert_zero-shot-classification_chinese-base",
        "launcher": "xinference.model.flexible.launchers.modelscope",
        "launcher_args": "{\"task\": \"zero-shot-classification\"}",
        "virtualenv": {
            "packages": [
                "transformers==4.50.3",
                "#system_numpy#"
            ],
            "inherit_pip_config": true,
            "index_url": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple",
            "extra_index_url": null,
            "find_links": null,
            "trusted_host": null,
            "no_build_isolation": null
        },
        "is_builtin": false
    }

Inference the model:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/flexible/infers' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "nlp_structbert_zero-shot-classification_chinese-base",
        "args": [
          "世界那么大，我想去看看"
        ],
        "candidate_labels": ["家居", "旅游", "科技", "军事", "游戏", "故事"]
      }'

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("nlp_structbert_zero-shot-classification_chinese-base")

    labels = ['家居', '旅游', '科技', '军事', '游戏', '故事']
    sentence = '世界那么大，我想去看看'
    model.infer(sentence, candidate_labels=labels)


  .. code-tab:: json output

    {"labels":["旅游","故事","游戏","家居","科技","军事"],"scores":[0.5115892291069031,0.1660086065530777,0.11971458047628403,0.08431519567966461,0.06298774480819702,0.05538458004593849]}%