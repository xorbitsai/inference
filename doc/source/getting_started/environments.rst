.. _environments:

======================
Environments Variables
======================

XINFERENCE_ENDPOINT
~~~~~~~~~~~~~~~~~~~~
Endpoint of Xinference, used to connect to Xinference service.
Default value is http://127.0.0.1:9997 , you can get it through logs.

XINFERENCE_MODEL_SRC
~~~~~~~~~~~~~~~~~~~~~
Modelhub used for downloading models. Default is "huggingface", or you
can set "modelscope" as downloading source.

.. _environments_xinference_home:

XINFERENCE_HOME
~~~~~~~~~~~~~~~~
By default, Xinference uses ``<HOME>/.xinference`` as home path to store
necessary files such as logs and models, where ``<HOME>`` is the home
path of current user. You can change this directory by configuring this environment
variable.

XINFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The maximum number of failed health checks tolerated at Xinference startup.
Default value is 5.

XINFERENCE_HEALTH_CHECK_INTERVAL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Health check interval (seconds) at Xinference startup.
Default value is 5.

XINFERENCE_HEALTH_CHECK_TIMEOUT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Health check timeout (seconds) at Xinference startup.
Default value is 10.

XINFERENCE_DISABLE_HEALTH_CHECK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Xinference will automatically report health check at Xinference startup.
Setting this environment to 1 can disable health check.

XINFERENCE_DISABLE_METRICS
~~~~~~~~~~~~~~~~~~~~~~~~~~
Xinference will by default enable the metrics exporter on the supervisor and worker.
Setting this environment to 1 will disable the /metrics endpoint on the supervisor
and the HTTP service (only provide the /metrics endpoint) on the worker.

XINFERENCE_DOWNLOAD_MAX_ATTEMPTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maximum download retry attempts for model files.
Default value is 3.

XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enable continuous batching for text-to-image models by specifying the target image size
(e.g., ``1024*1024``). Default is unset.

XINFERENCE_SSE_PING_ATTEMPTS_SECONDS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Server-Sent Events keepalive ping interval (seconds).
Default value is 600.

XINFERENCE_MAX_TOKENS
~~~~~~~~~~~~~~~~~~~~~
Global max tokens limit override for requests. Default is unset.

XINFERENCE_ALLOWED_IPS
~~~~~~~~~~~~~~~~~~~~~~
Restrict access to specified IPs or CIDR blocks. Default is unset (no restriction).

XINFERENCE_BATCH_SIZE
~~~~~~~~~~~~~~~~~~~~~
Default batch size used by the server when batching is enabled.
Default value is 32.

XINFERENCE_BATCH_INTERVAL
~~~~~~~~~~~~~~~~~~~~~~~~~
Default batching interval (seconds).
Default value is 0.003.

XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Whether to allow multiple replicas on a single GPU.
Default value is 1 (enabled).

XINFERENCE_LAUNCH_STRATEGY
~~~~~~~~~~~~~~~~~~~~~~~~~~
GPU allocation strategy for replicas. Default is ``IDLE_FIRST_LAUNCH_STRATEGY``.

XINFERENCE_MAX_CONCURRENT_LAUNCHES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maximum number of model launches that can proceed concurrently on a single
worker node. When more replicas are launched than this limit, excess launches
queue and proceed as slots free up. This prevents resource exhaustion (fork
storms, disk IO saturation, GPU memory contention) that can cause heartbeat
timeouts. Default value is 5.

XINFERENCE_ENABLE_VIRTUAL_ENV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enable model virtual environments globally.
Default value is 1 (enabled, starting from v2.0).

XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Skip packages already present in system site-packages when creating virtual environments.
Default value is 1.

XINFERENCE_CSG_TOKEN
~~~~~~~~~~~~~~~~~~~~
Authentication token for CSGHub model source.
Default is unset.

XINFERENCE_CSG_ENDPOINT
~~~~~~~~~~~~~~~~~~~~~~~
CSGHub endpoint for model source.
Default value is ``https://hub-stg.opencsg.com/``.

XINFERENCE_QWEN3_RERANK_TEMPLATE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enable template for Qwen3 rerank model family (0.6B, 4B, 8B,etc) globally.
Default value is 1.

XINFERENCE_LAUNCH_HISTORY_DB_PATH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Path to the SQLite database that stores the model launch configuration history
shown in the "Launch Model" drawer of the Web UI. This store is shared across
all clients so the history is available from any browser or machine, and it is
independent of the authentication database. When authentication is enabled, each
record keeps the creator's username (``created_by``).
Default value is ``<XINFERENCE_HOME>/launch_history.db``.

XINFERENCE_MODEL_ACTOR_AUTO_RECOVER_LIMIT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maximum number of times a crashed model actor is automatically recovered.
Default is unset (no limit).

XINFERENCE_FRONTEND_DIST_DIR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Path to a static export of the web UI to serve instead of the one bundled
with the package. Default is unset (serve the bundled Next.js export).

.. versionchanged:: 3.0
   Replaces the removed ``XINFERENCE_FRONTEND_ENDPOINT`` variable; the web UI
   is now served by the Xinference server itself.

XINFERENCE_AUTH_ADVANCED
~~~~~~~~~~~~~~~~~~~~~~~~
Enable the database-backed authentication system. Default value is 1
(enabled, starting from v3.0). Set to ``0`` / ``false`` / ``no`` to run
Xinference without any authentication.
See :ref:`user_guide_auth_system`.

XINFERENCE_AUTH_DB_PATH
~~~~~~~~~~~~~~~~~~~~~~~
Path to the SQLite database that stores users, permissions, API keys, and
refresh tokens. Default value is ``<XINFERENCE_HOME>/auth/auth.db``.

XINFERENCE_AUTH_JWT_SECRET_KEY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
JWT signing secret for the authentication system. If unset, a secret is
auto-generated and persisted at ``<XINFERENCE_HOME>/auth/jwt_secret_key`` on
first run.

XINFERENCE_AUTH_ENCRYPTION_KEY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Key used to encrypt stored API keys at rest. If unset, a key is
auto-generated and persisted at ``<XINFERENCE_HOME>/auth/encryption_key`` on
first run.

XINFERENCE_ACCESS_TOKEN_EXPIRE_MINUTES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lifetime (minutes) of access tokens issued by the authentication system.
Default value is 30.

XINFERENCE_OIDC_ENABLED
~~~~~~~~~~~~~~~~~~~~~~~
Enable OIDC single sign-on (e.g. Keycloak). Default value is 0 (disabled).
When enabled, ``XINFERENCE_OIDC_ISSUER``, ``XINFERENCE_OIDC_CLIENT_ID``,
``XINFERENCE_OIDC_CLIENT_SECRET``, and ``XINFERENCE_OIDC_REDIRECT_URI`` are
required. See :ref:`user_guide_oidc`.

XINFERENCE_AUDIT_LOG_RETENTION_DAYS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Number of days audit log files are retained.
Default value is 90. See :ref:`user_guide_audit_security`.
