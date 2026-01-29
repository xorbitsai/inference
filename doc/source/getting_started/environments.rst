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
