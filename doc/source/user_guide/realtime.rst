.. _realtime_sessions:

Realtime model sessions
=======================

Use ``WS /v1/models/{model_uid}/realtime`` for continuous media input. When authentication is enabled, send ``Authorization: Bearer <API_KEY>``; model permissions and the ``models:read`` scope apply. This is a Xinference protocol, not the OpenAI Realtime API.

The configured ``XINFERENCE_ALLOWED_IPS`` allowlist also applies before WebSocket upgrade.

The initial adapter is opt-in segmental transcription for PyTorch ``Fun-ASR-Nano-2512``. Launch with ``realtime=True`` and a FunASR version providing ``DynamicStreamingVAD``. It reuses loaded ASR and FSMN VAD weights; it does not load vLLM or promise incremental acoustic encoding. Other instances remain unchanged.

.. code-block:: python

   from xinference.client import Client

   client = Client("http://localhost:9997", api_key="<API_KEY>")
   uid = client.launch_model(
       model_name="Fun-ASR-Nano-2512",
       model_type="audio",
       model_engine="PyTorch",
       realtime=True,
   )

Send the configuration below, wait for ``session.started``, then send binary mono 16 kHz PCM16 little-endian audio without a WAV header. Send ``input.end`` to flush accepted audio, or ``session.cancel`` to cancel. Results use increasing ``seq`` values; replace cumulative ``sentences`` and mutable ``partial`` snapshots instead of appending them. Success requires one final output and ``session.closed`` with reason ``completed``.

.. code-block:: json

   {
     "event": "session.start",
     "protocol_version": 1,
     "task": "audio.transcription",
     "input": {
       "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
     },
     "parameters": {"language": "zh", "hotwords": []}
   }

Only connection initialization uses Actor routing. Media and results use an authenticated persistent connection to the model process. Configure ``XINFERENCE_REALTIME_BIND_HOST`` and ``XINFERENCE_REALTIME_ADVERTISE_HOST`` when the Worker address is unsuitable. Keep internal ports on a trusted private network or encrypted tunnel; this endpoint does not provide TLS itself. Browser origins are rejected unless listed in ``XINFERENCE_REALTIME_ALLOWED_ORIGINS``; browser-native ticket authentication is not implemented.

For the Nano adapter, the limits and locking below apply. The Qwen3 engines use the separate defaults described later.

Defaults are one realtime session per replica, 64 KiB input frames, 2 MiB output messages, eight-element input and output queues, 60 seconds of media inactivity, and 30 minutes per session. File requests and realtime sessions share request-limit accounting and inference locking. Cancellation retains capacity until an already running native operation finishes; safe unloading can therefore exceed the drain timeout. Real-GPU latency, accuracy and capacity still require deployment validation.

.. code-block:: bash

   export XINFERENCE_REALTIME_MAX_SESSIONS=1

Run the repository probe with a mono PCM16 WAV, using ``XINFERENCE_API_KEY`` from the environment. A 30–60 second speech recording must produce nonempty text before the last audio frame and ``input.end``. The probe cannot independently prove zero media RPC or resource cleanup; run the runtime integration tests as well.

.. code-block:: bash

   python benchmark/realtime_ws_probe.py \
     --url "ws://localhost:9997/v1/models/<MODEL_UID>/realtime" \
     --audio ./samples/zh_60s.wav \
     --chunk-ms 100 --require-interim \
     --report ./reports/nano_realtime.json

Nano launch options
-------------------

With ``realtime=True``, set ``realtime_options`` at model launch, not in session ``parameters``. In the Web UI, add an engine parameter named ``realtime_options`` with the JSON value below; the UI submits it as an object. REST/Python clients should send an object; JSON object strings are accepted for compatibility with older Web UIs. Unknown fields, non-object JSON and invalid value types are rejected.

.. code-block:: json

   {"optimize":true,"first_partial_ms":960,"partial_interval_ms":1500,"max_segment_ms":6000}

Optimization is disabled by default. This is a candidate configuration, not a latency guarantee. Removing ``realtime_options`` restores defaults; setting only ``optimize=false`` does not reset explicitly configured timing or segment limits.


Qwen3 audio with native batching
--------------------------------

Query ``GET /v1/models/{model_uid}/audio/capabilities`` before enabling voice controls. It reports the loaded adapter’s streaming and VAD support, supported TTS formats, sample rates and voice IDs. Unsupported adapters return an empty object. ``formats`` and ``sample_rates`` describe streaming output; ``file_formats`` lists non-streaming output formats.

Use Linux with CUDA and Python 3.11–3.13. In the model environment, install ``vllm[audio]==0.32.0`` and ``silero-vad[onnx-cpu]`` for ASR; TTS additionally requires ``vllm-omni==0.32.0``. Engine-specific virtualenv packages are included in the built-in model specs.

Launch Qwen3-ASR with ``vLLM``. Each connection owns a Silero VAD state and utterance buffer; English is the default session language. Use the WebSocket protocol above with ``language=en``. Partial results re-decode utterance snapshots; this is segmental transcription, not incremental acoustic encoding. Both 0.6B and 1.7B models are supported.

.. code-block:: python

   asr_uid = client.launch_model(
       model_name="Qwen3-ASR-0.6B", model_type="audio",
       model_engine="vLLM", max_num_seqs=16,
       realtime_options={"partial_interval_ms": 1000, "silence_ms": 512,
                         "min_speech_ms": 160, "max_segment_ms": 6000},
   )

ASR defaults to 16 concurrent sessions per replica. ``XINFERENCE_REALTIME_MAX_SESSIONS`` overrides admission capacity; ``max_num_seqs`` configures native engine scheduling. Concurrent file and WebSocket decodes enter the same AsyncLLM scheduler without a Python inference lock. Cancelling one session aborts only its request and retains its slot until abort settles. Measure throughput, latency and accuracy on the target GPU before choosing production limits.

For ASR, ``max_num_seqs`` must be 1–128 (default 16); ``language=auto`` enables automatic detection. Native engine construction currently blocks the model Actor loop during loading; RPCs to that replica resume after initialization.

Launch Qwen3-TTS CustomVoice with ``vLLM`` for native AsyncOmni batching and streaming. Both 0.6B and 1.7B CustomVoice models are supported; Base and VoiceDesign are excluded. Streaming requires ``stream=true`` and ``response_format=pcm`` and returns mono 24 kHz PCM16 little-endian chunks. The default speaker is ``ryan`` and the default language is English. Close the response to cancel generation.

.. code-block:: python

   tts_uid = client.launch_model(
       model_name="Qwen3-TTS-12Hz-0.6B-CustomVoice", model_type="audio",
       model_engine="vLLM", max_num_seqs=16,
   )

.. code-block:: bash

   curl --no-buffer http://localhost:9997/v1/audio/speech \
     -H "Authorization: Bearer $XINFERENCE_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"model":"<TTS_UID>","input":"Hello. How can I help?","voice":"ryan","response_format":"pcm","stream":true}' \
     --output speech.pcm
