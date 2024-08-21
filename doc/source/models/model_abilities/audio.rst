.. _audio:

=====================
Audio (Experimental)
=====================

Learn how to turn audio into text or text into audio with Xinference.


Introduction
==================


The Audio API provides three methods for interacting with audio:


* The transcriptions endpoint transcribes audio into the input language.
* The translations endpoint translates audio into English.
* The speech endpoint generates audio from the input text.


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

Audio to text
~~~~~~~~~~~~~

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
* SenseVoiceSmall


Text to audio
~~~~~~~~~~~~~

* ChatTTS
* CosyVoice

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

.. _audio_speech:

The Speech API mimics OpenAI's `create speech API <https://platform.openai.com/docs/api-reference/audio/createSpeech>`_.
We can try Speech API out either via cURL, OpenAI Client, or Xinference's python client:

Speech API use non-stream by default as

1. The stream output of ChatTTS is not as good as the non-stream output, please refer to: https://github.com/2noise/ChatTTS/pull/564
2. The stream requires ffmpeg<7: https://pytorch.org/audio/stable/installation.html#optional-dependencies

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/audio/speech' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "input": "<The text to generate audio for>",
        "voice": "echo",
        "stream": True,
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
        voice="echo",
        stream: True,
    )


  .. code-tab:: output

    The output will be an audio binary.


ChatTTS Usage
~~~~~~~~~~~~~

Basic usage, refer to :ref:`audio speech usage <audio_speech>`.

Fixed tone color. We can use fixed tone color provided by
https://github.com/6drf21e/ChatTTS_Speaker,
Download the `evaluation_result.csv <https://github.com/6drf21e/ChatTTS_Speaker/blob/main/evaluation_results.csv>`_ ,
take ``seed_2155`` as example, we get the ``emb_data`` of it.

.. code-block:: python

    import pandas as pd

    df = pd.read_csv("evaluation_results.csv")
    emb_data_2155 = df[df['seed_id'] == 'seed_2155'].iloc[0]["emb_data"]


Use the fixed tone color of ``seed_2155`` to generate speech.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    resp_bytes = model.speech(
        voice=emb_data_2155,
        input=<The text to generate audio for>
    )


CosyVoice Usage
~~~~~~~~~~~~~~~

Basic usage, launch model ``CosyVoice-300M-SFT``.

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/audio/speech' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "input": "<The text to generate audio for>",
        # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
        "voice": "中文女"
      }'

  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
        api_key="cannot be empty",
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    response = client.audio.speech.create(
        model=<MODEL_UID>,
        input=<The text to generate audio for>,
        # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
        voice="中文女",
    )
    response.stream_to_file('1.mp3')

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    speech_bytes = model.speech(
        input=<The text to generate audio for>,
        # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
        voice="中文女"
    )
    with open('1.mp3', 'wb') as f:
        f.write(speech_bytes)


Clone voice, launch model ``CosyVoice-300M``.

.. code-block::

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")

    zero_shot_prompt_text = ""
    # The zero shot prompt file is the voice file
    # the words said in the file shoule be identical to zero_shot_prompt_text
    with open(zero_shot_prompt_file, "rb") as f:
        zero_shot_prompt = f.read()

    speech_bytes = model.speech(
        "<The text to generate audio for>",
        prompt_text=zero_shot_prompt_text,
        prompt_speech=zero_shot_prompt,
    )


Cross lingual usage, launch model ``CosyVoice-300M``.

.. code-block::

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")

    # the file that reads in some language
    with open(cross_lingual_prompt_file, "rb") as f:
        cross_lingual_prompt = f.read()

    speech_bytes = model.speech(
        "<The text to generate audio for>",  # text could be another language
        prompt_speech=cross_lingual_prompt,
    )

Instruction based, launch model ``CosyVoice-300M-Instruct``.

.. code-block::

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")

    response = model.speech(
        "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。",
        voice="中文男",
        instruct_text="Theo 'Crimson', is a fiery, passionate rebel leader. "
        "Fights with fervor for justice, but struggles with impulsiveness.",
    )

More instructions and examples, could be found at https://fun-audio-llm.github.io/ .
