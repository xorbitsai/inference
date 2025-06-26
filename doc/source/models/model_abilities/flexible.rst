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

Traditional machine learning models can still play a significant role within an LLM-centric ecosystem.

Xinference provides flexible extensibility for performing inference with traditional machine learning models.
It includes built-in support for loading and running the following types of models:

- Hugging Face Pipelines for tasks such as classification using models hosted on Hugging Face.
- ModelScope Pipelines for tasks such as classification using models from ModelScope.
- YOLO for image detection and related computer vision tasks.

A wide range of traditional machine learning models can be used with Xinference.
For each of the categories above, we will walk through a representative example to
demonstrate how to perform inference step by step on the Xinference platform.

Built-in Model Support Examples
================================

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

    <img class="align-center" alt="actor" src="../../_static/hf-pipeline.png" style="background-color: transparent", width="95%">

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

When using the command line, remember to specify the option ``--model-type flexible``.

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
This is because the model used in this example requires ``transformers==4.50.3`` to run properly.
To isolate the environment, we use a :ref:`virtual env <model_virtual_env>` when registering the model.

When specifying custom packages during registration, the syntax is the same as for regular packages, with a few special cases.
Since the virtual environment is still based on the site packages of the Python runtime where Xinference is running, we need to explicitly include `#system_numpy#`.
Packages wrapped in ``#system_xx#`` ensure consistency with the base environment during virtual environment creation; otherwise, it may easily result in runtime errors.

Registering via Web UI:

.. raw:: html

    <img class="align-center" alt="actor" src="../../_static/modelscope-pipeline.png" style="background-color: transparent", width="95%">

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

YOLO
~~~~

YOLO is a popular real-time object detection model, widely used in image detection and video analysis scenarios.

First, download the YOLO weights.
Here, we use the `yolov11s.pt <https://huggingface.co/Ultralytics/YOLO11>`_ file as an example.

JSON file of model definition:

.. code-block:: json

    {
        "model_name": "yolo11s",
        "model_id": null,
        "model_revision": null,
        "model_hub": "huggingface",
        "model_description": "YOLO is a popular real-time object detection model, widely used in image detection and video analysis scenarios.",
        "model_uri": "/Users/xuyeqin/Downloads/models/yolo11s.pt",
        "launcher": "xinference.model.flexible.launchers.yolo",
        "launcher_args": "{}",
        "virtualenv": {
            "packages": [
                "ultralytics",
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

  .. code-tab:: python Xinference Python Client

    import requests
    from PIL import Image
    import io
    import base64
    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model("yolo11s")

    url = "https://ultralytics.com/images/bus.jpg"

    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(io.BytesIO(response.content))

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    model.infer(source=img_base64)

  .. code-tab:: json output

    [[{'name': 'bus',
       'class': 5,
       'confidence': 0.93653,
       'box': {'x1': 13.9521, 'y1': 227.0665, 'x2': 800.17688, 'y2': 739.13965}},
      {'name': 'person',
       'class': 0,
       'confidence': 0.89741,
       'box': {'x1': 669.89709,
        'y1': 389.82065,
        'x2': 809.58966,
        'y2': 879.65491}},
      {'name': 'person',
       'class': 0,
       'confidence': 0.88205,
       'box': {'x1': 52.37262, 'y1': 397.83792, 'x2': 248.506, 'y2': 905.98212}},
      {'name': 'person',
       'class': 0,
       'confidence': 0.8706,
       'box': {'x1': 222.58685,
        'y1': 405.93442,
        'x2': 345.02032,
        'y2': 859.52789}},
      {'name': 'person',
       'class': 0,
       'confidence': 0.66505,
       'box': {'x1': 0.28522, 'y1': 548.60931, 'x2': 81.25904, 'y2': 871.59076}}]]

Writing a Custom Flexible Model
==================================

First, we implement a custom launcher with a simple model for sentiment scoring.
In this example, we do not use any actual model weights, so the ``load`` function does not perform any model loading.

.. code-block:: python

    # my_flexible_model.py

    from xinference.model.flexible import FlexibleModel


    class RuleBasedSentimentModel(FlexibleModel):
        def load(self):
            self.pos_words = self.config.get("pos", ["good", "happy", "great"])
            self.neg_words = self.config.get("neg", ["bad", "sad", "terrible"])

        def infer(self, text: str):
            score = 0
            words = text.lower().split()
            for w in words:
                if w in self.pos_words:
                    score += 1
                elif w in self.neg_words:
                    score -= 1
            return {"score": score}


    def launcher(model_uid: str, model_spec: FlexibleModel, **kwargs) -> FlexibleModel:
        # get model path,
        # in this example, we do not use it, so it's empty
        model_path = model_spec.model_uri
        return RuleBasedSentimentModel(model_uid=model_uid, model_path=model_path, config=kwargs)

The model JSON definition is as follows:

.. code-block:: json

    {
        "model_name": "my-flexible-model",
        "model_id": null,
        "model_revision": null,
        "model_hub": "huggingface",
        "model_description": "This is a model description.",
        "model_uri": "/path/to/model",
        "launcher": "my_flexible_model.launcher",
        "launcher_args": "{\"pos\": [\"good\", \"happy\", \"great\", \"nice\"]}",
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

Here, we extend the model by passing in a custom-defined ``pos`` value.

Finally, let's verify the result:

.. tabs::

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")

    model = client.get_model("my-flexible-model")

    model.infer("I feel nice and am happy today")

  .. code-tab:: json output

    {'score': 2}

Conclusion
==================

The built-in Flexible Model launchers in Xinference can be found at
`Github <https://github.com/xorbitsai/inference/tree/main/xinference/model/flexible/launchers>`_.
Contributions are welcome to support more traditional machine learning models!
