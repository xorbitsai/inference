.. _models_builtin_fireredtts3:

===========
FireRedTTS3
===========

Xinference exposes the two checkpoints in the upstream repository as separate
model cards. Both cards use ``FireRedTeam/FireRedTTS3`` and share one local
cache. Hugging Face uses revision ``main``; ModelScope uses revision ``master``.
Both variants require a CUDA-capable GPU.

FireRedTTS3-Base
^^^^^^^^^^^^^^^^

- **Model Name:** FireRedTTS3-Base
- **Abilities:** ['text2audio', 'text2audio_voice_cloning']
- **Multilingual:** True

Base provides multilingual zero-shot voice cloning across 24 languages and 21
Chinese dialects. It requires reference audio and the corresponding transcript.

.. code-block:: bash

   xinference launch --model-name FireRedTTS3-Base --model-type audio

.. code-block:: python

   from xinference.client import Client

   client = Client("http://127.0.0.1:9997")
   model = client.get_model("FireRedTTS3-Base")

   with open("prompt.wav", "rb") as f:
       prompt_speech = f.read()

   audio = model.speech(
       input="今天天气很好，我们一起去公园散步吧。",
       prompt_speech=prompt_speech,
       prompt_text="这里填写参考音频对应的完整文本。",
       language="Chinese",
       response_format="wav",
   )

FireRedTTS3-Instruct
^^^^^^^^^^^^^^^^^^^^

- **Model Name:** FireRedTTS3-Instruct
- **Abilities:** ['text2audio', 'text2audio_voice_design',
  'text2audio_voice_cloning']
- **Multilingual:** True

Instruct supports voice design without reference audio. To reuse Xinference's
existing speech request fields, pass the natural-language voice description in
``prompt_text`` and omit ``prompt_speech``.

.. code-block:: bash

   xinference launch --model-name FireRedTTS3-Instruct --model-type audio

.. code-block:: python

   model = client.get_model("FireRedTTS3-Instruct")

   audio = model.speech(
       input="今天天气很好，我们一起去公园散步吧。",
       prompt_text="一个年轻女性的温柔嗓音，语速稍慢，带一点俏皮。",
       response_format="wav",
   )

Instruct also supports zero-shot voice cloning. When ``prompt_speech`` is
provided, ``prompt_text`` changes back to its standard meaning: the reference
audio transcript.

The upstream semantic and acoustic speech-editing methods are not exposed by
Xinference yet. They require audio-to-audio abilities and request fields beyond
the standard speech API.

The optional upstream inference parameters ``stop_threshold``, ``n_timesteps``,
``inference_cfg``, and ``seed`` can be passed to ``speech``. Streaming and the
OpenAI ``speed`` control are not supported.
