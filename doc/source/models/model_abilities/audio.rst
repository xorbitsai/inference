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
* whisper-large-v3-turbo
* Belle-distilwhisper-large-v2-zh
* Belle-whisper-large-v2-zh
* Belle-whisper-large-v3-zh
* SenseVoiceSmall


Text to audio
~~~~~~~~~~~~~

* ChatTTS
* CosyVoice
* FishSpeech-1.5
* F5-TTS

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
    # the words said in the file should be identical to zero_shot_prompt_text
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

CosyVoice 2.0 usage, launch model ``CosyVoice2-0.5B``.

.. note::

    Please note that the latest CosyVoice 2.0 requires `use_flow_cache=True` for stream generation.

.. code-block::

    from xinference.client import Client

    model_uid = client.launch_model(
        model_name=model_name,
        model_type="audio",
        download_hub="modelscope",
        use_flow_cache=True,
    )


More instructions and examples, could be found at https://fun-audio-llm.github.io/ .


FishSpeech Usage
~~~~~~~~~~~~~~~~

Basic usage, refer to :ref:`audio speech usage <audio_speech>`.

Clone voice, launch model ``FishSpeech-1.5``. Please use `prompt_speech` instead of `reference_audio`
and `prompt_text` instead of `reference_text` to clone voice from the reference audio for the FishSpeech model.
This arguments is aligned to voice cloning of CosyVoice.

.. code-block::

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")

    # The reference audio file is the voice file
    # the words said in the file should be identical to reference_text
    with open(reference_audio_file, "rb") as f:
        reference_audio = f.read()
    reference_text = ""  # text in the audio

    speech_bytes = model.speech(
        "<The text to generate audio for>",
        prompt_speech=reference_audio,
        prompt_text=reference_text
    )


SenseVoiceSmall Offline Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now SenseVoiceSmall use a small vad model ``fsmn-vad``, it will be downloaded thus network required.

For offline environment, you can download the vad model in advance.

Download from `huggingface <https://huggingface.co/funasr/fsmn-vad>`_ or `modelscope <https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/files>`_.
Assume downloaded to ``/path/to/fsmn-vad``.

Then when launching SenseVoiceSmall with Web UI, you can add an additional parameter with key ``vad_model`` and value ``/path/to/fsmn-vad`` which is the downloaded path.
When launching with command line, you can add an option ``--vad_model /path/to/fsmn-vad``.


Kokoro Usage
~~~~~~~~~~~~

The Kokoro model supports multiple languages, but the default language is English.
If you want to use other languages, such as Chinese, you need to install additional dependency packages
and add an additional parameter when starting the model.

1. pip install misaki[zh]

2. Initialize the model with the parameter lang_code='z',
   For all available ``lang_code`` options,
   please refer to `kokoro source code <https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py#L22>`_.
   If the model is started through the web UI, an additional
   parameter needs to be added, with the key as ``lang_code`` and the value as ``z``.
   If the model is started through the xinference client, the parameters are passed via the launch_model interface:

   .. code-block::

       model_uid = client.launch_model(
           model_name="Kokoro-82M",
           model_type="audio",
           compile=False,
           download_hub="huggingface",
           lang_code="z",
       )

3. When inferring, the voice must start with 'z', for example: ``zf_xiaoyi``.
   The currently supported voices are: https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices. For example:

   .. code-block::

       input_string = "重新启动即可更新"
       response = model.speech(input_string, voice="zf_xiaoyi")
