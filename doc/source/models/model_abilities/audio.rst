.. _audio:

=====
Audio
=====

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

* :ref:`whisper-tiny <models_builtin_whisper-tiny>`
* :ref:`whisper-tiny.en <models_builtin_whisper-tiny.en>`
* :ref:`whisper-base <models_builtin_whisper-base>`
* :ref:`whisper-base.en <models_builtin_whisper-base.en>`
* :ref:`whisper-medium <models_builtin_whisper-medium>`
* :ref:`whisper-medium.en <models_builtin_whisper-medium.en>`
* :ref:`whisper-large-v3 <models_builtin_whisper-large-v3>`
* :ref:`whisper-large-v3-turbo <models_builtin_whisper-large-v3-turbo>`
* :ref:`Belle-distilwhisper-large-v2-zh <models_builtin_belle-distilwhisper-large-v2-zh>`
* :ref:`Belle-whisper-large-v2-zh <models_builtin_belle-whisper-large-v2-zh>`
* :ref:`Belle-whisper-large-v3-zh <models_builtin_belle-whisper-large-v3-zh>`
* :ref:`SenseVoiceSmall <models_builtin_sensevoicesmall>`
* :ref:`Paraformer-zh <models_builtin_paraformer-zh>`

For Mac M-series chips only:

* :ref:`whisper-tiny-mlx <models_builtin_whisper-tiny-mlx>`
* :ref:`whisper-tiny.en-mlx <models_builtin_whisper-tiny.en-mlx>`
* :ref:`whisper-base-mlx <models_builtin_whisper-base-mlx>`
* :ref:`whisper-base.en-mlx <models_builtin_whisper-base.en-mlx>`
* :ref:`whisper-medium-mlx <models_builtin_whisper-medium-mlx>`
* :ref:`whisper-medium.en-mlx <models_builtin_whisper-medium.en-mlx>`
* :ref:`whisper-large-v3-mlx <models_builtin_whisper-large-v3-mlx>`
* :ref:`whisper-large-v3-turbo-mlx <models_builtin_whisper-large-v3-turbo-mlx>`


Text to audio
~~~~~~~~~~~~~

* :ref:`ChatTTS <models_builtin_chattts>`
* :ref:`CosyVoice-300M-SFT <models_builtin_cosyvoice-300m-sft>`
* :ref:`CosyVoice-300M <models_builtin_cosyvoice-300m>`
* :ref:`CosyVoice-300M-Instruct <models_builtin_cosyvoice-300m-instruct>`
* :ref:`CosyVoice 2.0 <models_builtin_cosyvoice2-0.5b>`
* :ref:`FishSpeech-1.5 <models_builtin_fishspeech-1.5>`
* :ref:`F5-TTS <models_builtin_f5-tts>`
* :ref:`MegaTTS3 <models_builtin_megatts3>`
* MeloTTS series

For Mac M-series chips only:

* :ref:`F5-TTS-MLX <models_builtin_f5-tts-mlx>`

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

CosyVoice has two versions: CosyVoice 1.0 and CosyVoice 2.0. CosyVoice 1.0 has three different models:

- **CosyVoice-300M-SFT**: Choose this model if you just want to convert text to audio. There are pretrained voices available: ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
- **CosyVoice-300M**: Choose this model if you want to clone voice or convert text to audio in different languages. The ``prompt_speech`` is always required and should be a WAV file. For optimal performance, use a sample rate of 16,000 Hz.
- **CosyVoice-300M-Instruct**: Choose this model If you need precise control over the tone and pitch.

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

    zero_shot_prompt_text = ("<the words in the text exactly match "
                             "the audio file of the zero-shot prompt>")
    # The words said in the audio file should be identical
    # to zero_shot_prompt_text.
    #
    # The audio input file must be in WAV format.
    # For optimal performance, use a 16,000 Hz sample rate.
    #
    # Files with different sample rates will be resampled to 16,000 Hz,
    # which may increase processing time.
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

    # The audio input file must be in WAV format.
    # For optimal performance, use a 16,000 Hz sample rate.
    #
    # Files with different sample rates will be resampled to 16,000 Hz,
    # which may increase processing time.
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

CosyVoice 2.0 only has one model, it provides all the capabilities of the three CosyVoice models. The usage is the same as CosyVoice, with the only difference being that CosyVoice 2.0 requires ``use_flow_cache=True`` when launching the model for stream generation.

CosyVoice 2.0 stream usage, launch model ``CosyVoice2-0.5B``.

.. note::

    Please note that the latest CosyVoice 2.0 requires `use_flow_cache=True` for stream generation.

.. code-block::

    # Launch model
    from xinference.client import Client

    model_uid = client.launch_model(
        model_name=model_name,
        model_type="audio",
        download_hub="modelscope",
        use_flow_cache=True,
    )

    endpoint = "http://127.0.0.1:9997"
    input_string = "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"

    # Stream request by openai client
    import openai
    import tempfile

    openai_client = openai.Client(api_key="not empty", base_url=f"{endpoint}/v1")
    # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    response = openai_client.audio.speech.with_streaming_response.create(
        model=model_uid, input=input_string, voice="英文女"
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        response.stream_to_file(f.name)
        assert os.stat(f.name).st_size > 0

    # Stream request by xinference client
    response = model.speech(input_string, stream=True)
    assert inspect.isgenerator(response)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        for chunk in response:
            f.write(chunk)


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
