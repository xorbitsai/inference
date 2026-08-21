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
All the logs are stored in the ``<XINFERENCE_HOME>/logs`` directory, where ``<XINFERENCE_HOME>`` can be configured as mentioned in :ref:`using_xinference`.

Xinference creates a subdirectory under the log directory ``<XINFERENCE_HOME>/logs``.
The name of the subdirectory corresponds to the Xinference cluster startup time in milliseconds.

Local deployment
================
In a local deployment, the logs of Xinference supervisor and Xinference workers are combined into a single file. An example of the log directory structure is shown below::

    <XINFERENCE_HOME>/logs
        └── local_1699503558105
            └── xinference.log

where ``1699503558105`` is the timestamp when the Xinference cluster was created.
Therefore, when you create a cluster locally multiple times, you can look for the corresponding logs based on this timestamp.

Distributed deployment
======================
In a distributed deployment, Xinference supervisor and Xinference workers each create their own subdirectory under the log directory.
The name of the subdirectory starts with the role name, followed by the role startup time in milliseconds.
An example of the log directory structure is shown below::

    <XINFERENCE_HOME>/logs
        └── supervisor_1699503558908
            └── xinference.log
            worker_1699503559105
            └── xinference.log


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
