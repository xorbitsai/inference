.. _realtime_sessions:

Realtime model sessions
=======================

Use ``WS /v1/models/{model_uid}/realtime`` for continuous media input. When authentication is enabled, send ``Authorization: Bearer <API_KEY>``; model permissions and the ``models:read`` scope apply. This is a Xinference protocol, not the OpenAI Realtime API.

A model must provide a realtime adapter; instances without one reject realtime connections. Task-specific input and output schemas are defined by each adapter.

Only connection initialization uses Actor routing. Media and results use an authenticated persistent connection to the model process. Configure ``XINFERENCE_REALTIME_BIND_HOST`` and ``XINFERENCE_REALTIME_ADVERTISE_HOST`` when the Worker address is unsuitable. Keep internal ports on a trusted private network or encrypted tunnel; this endpoint does not provide TLS itself. Browser origins are rejected unless listed in ``XINFERENCE_REALTIME_ALLOWED_ORIGINS``; browser-native ticket authentication is not implemented.

Defaults are one realtime session per replica, 64 KiB input frames, 2 MiB output messages, eight-element input and output queues, 60 seconds of media inactivity, and 30 minutes per session. File requests and realtime sessions share request-limit accounting and inference locking. Cancellation retains capacity until an already running native operation finishes; safe unloading can therefore exceed the drain timeout. Real-GPU latency, accuracy and capacity still require deployment validation.

.. code-block:: bash

   export XINFERENCE_REALTIME_MAX_SESSIONS=1
