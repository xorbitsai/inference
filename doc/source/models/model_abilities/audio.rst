.. _audio:

=====================
Audio (Experimental)
=====================

Learn how to turn audio into text or text into audio with Xinference.


Introduction
==================


The Audio API provides two methods for interacting with audio:


* The transcriptions endpoint transcribes audio into the input language.
* The translations endpoint translates audio into English.


.. list-table:: 
   :widths: 25  50
   :header-rows: 1

   * - API ENDPOINT
     - OpenAI-compatible ENDPOINT

   * - Transcription API
     - /v1/audio/transcriptions

   * - Translation API
     - /v1/audio/translations

   * - Speech API
     - /v1/audio/speech


Supported models
-------------------

The audio API is supported with the following models in Xinference:

* whisper-tiny
* whisper-tiny.en
* whisper-base
* whisper-base.en
* whisper-medium
* whisper-medium.en
* whisper-large-v3
* Belle-distilwhisper-large-v2-zh
* Belle-whisper-large-v2-zh
* Belle-whisper-large-v3-zh
* ChatTTS

Quickstart
===================

Transcription
--------------------

The Transcription API mimics OpenAI's `create transcriptions API <https://platform.openai.com/docs/api-reference/audio/createTranscription>`_.
We can try Transcription API out either via cURL, OpenAI Client, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/audio/transcriptions' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "file": "<audio bytes>",
      }'


  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
        api_key="cannot be empty", 
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    with open("speech.mp3", "rb") as audio_file:
        client.audio.transcriptions.create(
            model=<MODEL_UID>,
            file=audio_file,
        )

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    with open("speech.mp3", "rb") as audio_file:
        model.transcriptions(audio=audio_file.read())


  .. code-tab:: json output

    {
      "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that."
    }



Translation
--------------------

The Translation API mimics OpenAI's `create translations API <https://platform.openai.com/docs/api-reference/audio/createTranslation>`_.
We can try Translation API out either via cURL, OpenAI Client, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/audio/translations' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "file": "<audio bytes>",
      }'


  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
        api_key="cannot be empty",
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    with open("speech.mp3", "rb") as audio_file:
        client.audio.translations.create(
            model=<MODEL_UID>,
            file=audio_file,
        )

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    with open("speech.mp3", "rb") as audio_file:
        model.translations(audio=audio_file.read())


  .. code-tab:: json output

    {
      "text": "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?"
    }


Speech
--------------------

The Speech API mimics OpenAI's `create speech API <https://platform.openai.com/docs/api-reference/audio/createSpeech>`_.
We can try Speech API out either via cURL, OpenAI Client, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/audio/speech' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "text": "<The text to generate audio for>",
        "voice": "echo",
      }'


  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
        api_key="cannot be empty",
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    client.audio.speech.create(
        model=<MODEL_UID>,
        input=<The text to generate audio for>,
        voice="echo",
    )

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    model.speech(
        input=<The text to generate audio for>,
        voice="echo"
    )


  .. code-tab:: output

    The output will be an audio binary.
