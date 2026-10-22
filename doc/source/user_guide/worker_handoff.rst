.. _worker_hot_replacement:

Worker hot replacement
======================

Replace a distributed Worker on Linux or macOS while retaining its model processes. The Supervisor and API server keep running. Both Workers and the model processes must support the handoff protocol and use Xoscar 0.12.1 or later, compatible Python versions, the same Supervisor, GPU visibility and extensions. Local mode, models spanning Workers and Xavier or P/D deployments are currently unsupported.

To relaunch an existing Worker, send ``POST /v1/workers/{worker_address}/relaunch``. The Worker starts its replacement automatically and inherits its Supervisor, GPU visibility, runtime configuration and configured metrics exporter port; a new actor port is selected automatically.

.. code-block:: bash

    curl -X POST 'http://127.0.0.1:9997/v1/workers/127.0.0.1:9978/relaunch' \
      -H 'Content-Type: application/json' -d '{"request_id":"upgrade-1"}'

The response is HTTP 202 with an ``operation_id``. Poll ``GET /v1/worker-relaunches/{operation_id}`` or list recent operations with ``GET /v1/worker-relaunches``. Status is one of ``starting``, ``transferring``, ``pending``, ``succeeded`` or ``failed``. These routes require administrator access when authentication is enabled.

The Supervisor stores operation records in ``XINFERENCE_HOME/worker-relaunches.json``, so status remains available after the old Worker exits and can be reconciled after a Supervisor restart. Duplicate requests reuse an active operation. Keep the same optional ``request_id`` when retrying a request; use a new ID for a new operation. ``pending`` means the transfer outcome is uncertain: keep the processes alive and continue polling instead of starting another replacement.

The Supervisor also stores ownership reservations in ``XINFERENCE_HOME/worker-handoffs.json`` so an interrupted handoff can resume after a Supervisor restart. Replacement Workers keep updating the recovery file for the original Worker address, including models launched after relaunch.

The optional ``python_executable`` request field selects an absolute Python executable path on the Worker machine, for example a separately installed newer environment. It defaults to the current interpreter. Packages are not installed by this operation, and retained model processes keep their original environment and code.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_worker("127.0.0.1:9978")
    status = client.get_worker_relaunch(operation["operation_id"])

The CLI reports an error with guidance when ``--replace-worker`` would bypass the service host. Use the relaunch API for CLI-managed Workers and for unmanaged Workers started through the Python deployment API.

Model management changes pause during handoff; inference calls continue. Failure before ownership transfer restores the old Worker. After transfer starts, keep both processes alive while the replacement retries. The old Worker exits automatically after successful replacement.

The model processes retain their original Xinference and engine code and loaded weights. This replaces the Worker control process. Deploy both processes on the same machine, and configure the service manager to let retained processes outlive the old Worker instead of killing its process group or container.

Web UI
------

In **Cluster Information**, administrators can use **Restart / upgrade Worker** on a distributed Worker. Operation records remain visible when its address changes or the page is refreshed. This retains the current model processes; model-engine upgrades use **Reconfigure / upgrade model** in the running model's detail view.

Service and container ownership
-------------------------------

On Linux and macOS, ``xinference-worker`` automatically keeps a stable Worker host as its main process. Replacement Workers register before handoff, so retirement of the original Worker does not stop its models, replacement, systemd service or container. Explicitly stopping the host terminates its owned processes. Custom launch commands can use the host wrapper below.

.. code-block:: bash

    python -m xinference.deploy.worker_host xinference-worker \
      --endpoint http://127.0.0.1:9997 --host 0.0.0.0

A systemd service can keep its normal process-group cleanup, because the Worker host remains its main process:

.. code-block:: ini

    [Service]
    Type=simple
    ExecStart=/opt/xinference/bin/python -m xinference.deploy.worker_host /opt/xinference/bin/xinference-worker --endpoint http://127.0.0.1:9997 --host 0.0.0.0
    Restart=on-failure
    KillMode=control-group
    TimeoutStopSec=30

For Docker, use the same host command as the container command. Install or mount prepared Python environments inside the running container. Keep that container running during upgrades: replacing or stopping the container terminates its model processes and cannot preserve their GPU allocations.
