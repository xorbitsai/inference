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

Log Directory Structure
***********************
All the logs are stored in the ``<XINFERENCE_HOME>/logs`` directory, where ``<XINFERENCE_HOME>`` can be configured as mentioned in :ref:`using_xinference`.

Xinference creates a subdirectory under the log directory ``<XINFERENCE_HOME>/logs``.
The name of the subdirectory corresponds to the Xinference cluster startup time in milliseconds.

Local deployment
================
In a local deployment, the logs of Xinference supervisor and Xorbits workers are combined into a single file. An example of the log directory structure is shown below::

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
