.. _logging:

=====================
Logging in Xinference
=====================

Configure Log Level
###################
You can configure the log level with the ``--log-level`` option.
For example, starting a local cluster with ``DEBUG`` log level:

.. code-block:: bash

  xinference-local --log-level debug


Log Files
#########
Xinference supports log rotation of log files.
By default, logs rotate when they reach 100MB (maxBytes), and up to 30 backup files (backupCount) are kept.
Note that the log level configured above takes effect in both the command line logs and the log files.

Environment Variables
#####################
Xinference provides several environment variables to control logging behavior:

- ``XINFERENCE_LOG_CONSOLE``: Enable or disable console output (default: ``true``).
  When set to ``false``, logs are written only to files, and tqdm progress bars are captured and sampled.
- ``XINFERENCE_LOG_FORMAT``: Log format, either ``text`` (default) or ``json``.
- ``XINFERENCE_LOG_DOWNLOAD_PROGRESS``: Control how download progress bars are logged when ``XINFERENCE_LOG_CONSOLE=false``.
  Valid values are ``sampled`` (default, logs at 25/50/75/100% per file), ``full`` (logs every frame), or ``off`` (no progress logs).

Example usage:

.. code-block:: bash

  # Disable console output, log download progress at sampling points
  XINFERENCE_LOG_CONSOLE=false XINFERENCE_LOG_DOWNLOAD_PROGRESS=sampled xinference-local

  # Disable console output, log every download progress frame
  XINFERENCE_LOG_CONSOLE=false XINFERENCE_LOG_DOWNLOAD_PROGRESS=full xinference-local

  # Disable console output, no download progress logs
  XINFERENCE_LOG_CONSOLE=false XINFERENCE_LOG_DOWNLOAD_PROGRESS=off xinference-local

Log Directory Structure
***********************
Application logs are written to ``<XINFERENCE_LOG_DIR>/xinference.log``. By
default, ``XINFERENCE_LOG_DIR`` is ``<XINFERENCE_HOME>/logs``. If
``XINFERENCE_POD_NAME`` is set, the file is named
``xinference-<XINFERENCE_POD_NAME>.log`` instead. Local deployment combines
Supervisor and Worker logs in this file. In distributed deployment, each node
writes to its own local log directory. Rotated files stay beside the active file.

Web UI runtime logs
###################

Open **Monitoring Management > Log Center > Runtime logs** to view recent
application logs and errors without Elasticsearch. The page reads the active
log file on the selected node and refreshes every two seconds. It works when
Xinference runs directly with Python or inside Docker. If Elasticsearch is
configured, the Log Center also offers an Indexed logs tab for historical
search.

The runtime view shows the latest part of the active file and keeps a bounded
amount of text in the browser. Search filters only loaded text. Output written
directly to stdout or stderr outside Xinference logging, and failures before
the API starts, may require the terminal or ``docker logs``. Log access uses
the ``logs:list`` permission when authentication is enabled.


Token Router logging
####################

The independent ``xinference-router`` service uses the same Xinference file
formatters and rotation handlers as the Supervisor and Worker processes. A
production systemd deployment can use the following non-sensitive settings in
``/etc/xinference/router.env``:

.. code-block:: ini

  XINFERENCE_TOKEN_ROUTER_LOG_LEVEL=INFO
  XINFERENCE_TOKEN_ROUTER_ACCESS_LOG=false
  XINFERENCE_LOG_FORMAT=json
  XINFERENCE_LOG_CONSOLE=false
  XINFERENCE_LOG_DIR=/data/inference/logs/router
  XINFERENCE_LOG_ROTATION=daily+size
  XINFERENCE_LOG_RETENTION_DAYS=30
  XINFERENCE_LOG_MAX_BYTES=104857600
  XINFERENCE_LOG_BACKUP_COUNT=300

With these settings, Router application logs are written to
``/data/inference/logs/router/xinference.log``. The directory must exist and be
writable by the Router service account. Rotation is managed by Xinference; do
not apply an additional ``logrotate``/``copytruncate`` rule to the same file.

The Router emits structured lifecycle, configuration, routing decision,
completion, rejection, and backend-error events. Routing events include both
the requested virtual model and the selected physical backend model UID when
available. Request bodies, prompt/message content, response bodies,
Authorization headers, API keys, and control-plane tokens are not logged.

Uvicorn access logs are disabled by default because they duplicate high-volume
request information. Set ``XINFERENCE_TOKEN_ROUTER_ACCESS_LOG=true`` only when
access-log diagnostics are required. Uvicorn error logs remain enabled and use
the same Xinference logging configuration. In systemd deployments, journal
output should remain enabled as a fallback for process lifecycle messages and
failures that occur before application logging is initialized.

Model request body logging
##########################

Xinference can write inference request metadata and request bodies to a separate
JSON-lines file. This diagnostic feature is disabled by default because prompts,
media references, and other request values may contain sensitive data. Enable it
only on trusted systems with appropriate access controls and retention policies.
Authentication and authorization failures do not persist request bodies.
Multipart uploads record field values and original filenames, but not uploaded
file bytes. Requests larger than the configured capture limit, or requests
without a known size, record an omission reason instead of the body.
Each request event also records its producing process identity in ``role``,
``address``, ``node``, ``module``, and ``pid``, together with an
``api_protocol`` value of ``openai``, ``anthropic``, or ``xinference``.

The following environment variables configure the feature:

- ``XINFERENCE_MODEL_REQUEST_LOG_ENABLED``: Enable request body logging (default: ``false``).
- ``XINFERENCE_MODEL_REQUEST_LOG_FILE``: Log filename or absolute path (default: ``model_request.log``).
- ``XINFERENCE_MODEL_REQUEST_LOG_BODY_MAX_BYTES``: Maximum captured request size (default: ``16777216``). Set to ``-1`` to disable the size limit; this is not recommended for internet-facing deployments.
- ``XINFERENCE_MODEL_REQUEST_LOG_RETENTION_DAYS``: Maximum age of rotated files in days (default: ``7``).
- ``XINFERENCE_MODEL_REQUEST_LOG_MAX_BYTES``: Size-based rotation threshold (default: ``1073741824``).
- ``XINFERENCE_MODEL_REQUEST_LOG_BACKUP_COUNT``: Maximum number of rotated files (default: ``7``).

Each inference response includes ``X-Request-ID``. Xinference preserves a valid
caller-provided ``request-id`` or ``x-request-id`` value (in that order), or
generates an ``xinf-`` prefixed UUID. The same correlation ID is attached to
Supervisor model lookup logs without replacing model operation request IDs used
for cancellation or progress tracking.

For streaming responses, the terminal request log separates HTTP delivery from
stream execution with ``http_success``, ``stream_completed``,
``stream_outcome``, and, for non-successful outcomes, ``failure_origin``. A
response can therefore have HTTP status 200 while ``success`` and
``stream_completed`` are false. Endpoints report swallowed generator failures
through request state; the logging layer does not parse, buffer, or log stream
chunks. Failure origins are ``model_generator``, ``upstream``, ``protocol``,
``client``, or ``server``.

Correlation metadata is propagated independently across REST API, Supervisor,
Worker, ModelActor, PD, cancellation, and progress actor calls. Actor log records
can include ``correlation_id``, ``operation_request_id``, ``actor_call_id``, and
``parent_call_id``. The correlation ID is observability-only; the operation
request ID retains its existing cancellation, progress, batching, and backend
abort semantics. Xinference consumes the internal metadata envelope at actor
boundaries and does not pass it to model implementations.
