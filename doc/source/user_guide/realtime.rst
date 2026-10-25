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
