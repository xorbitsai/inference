.. _worker_hot_replacement:

Worker hot replacement
======================

Replace a distributed Worker on Linux or macOS while retaining its model processes. The Supervisor and API server keep running. Both Workers and the model processes must support the handoff protocol and use Xoscar 0.12.1 or later, compatible Python versions, the same Supervisor, GPU visibility and extensions. Local mode, models spanning Workers and Xavier or P/D deployments are currently unsupported.

To relaunch an existing Worker, send ``POST /v1/workers/{worker_address}/relaunch``. The Worker starts its replacement automatically and inherits its Supervisor, GPU visibility and runtime configuration; new actor and metrics ports are selected automatically.

.. code-block:: bash

    curl -X POST 'http://127.0.0.1:9997/v1/workers/127.0.0.1:9978/relaunch' \
      -H 'Content-Type: application/json' -d '{"request_id":"upgrade-1"}'

The response is HTTP 202 with an ``operation_id``. Poll ``GET /v1/worker-relaunches/{operation_id}`` or list recent operations with ``GET /v1/worker-relaunches``. Status is one of ``starting``, ``transferring``, ``pending``, ``succeeded`` or ``failed``. These routes require administrator access when authentication is enabled.

The Supervisor stores operation records in ``XINFERENCE_HOME/worker-relaunches.json``, so status remains available after the old Worker exits and can be reconciled after a Supervisor restart. Duplicate requests reuse an active operation. Keep the same optional ``request_id`` when retrying a request; use a new ID for a new operation. ``pending`` means the transfer outcome is uncertain: keep the processes alive and continue polling instead of starting another replacement.

The optional ``python_executable`` request field selects an absolute Python executable path on the Worker machine, for example a separately installed newer environment. It defaults to the current interpreter. Packages are not installed by this operation, and retained model processes keep their original environment and code.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_worker("127.0.0.1:9978")
    status = client.get_worker_relaunch(operation["operation_id"])

For an existing Worker at ``127.0.0.1:9978``, you can also start its replacement explicitly with a different actor port and a free metrics exporter port:

.. code-block:: bash

    xinference-worker -e http://127.0.0.1:9997 --worker-port 9998 \
      --replace-worker 127.0.0.1:9978 --metrics-exporter-port 19002

Model management changes pause during handoff; inference calls continue. Failure before ownership transfer restores the old Worker. After transfer starts, keep both processes alive while the replacement retries. The old Worker exits automatically after successful replacement.

The model processes retain their original Xinference and engine code and loaded weights. This replaces the Worker control process. Deploy both processes on the same machine, and configure the service manager to let retained processes outlive the old Worker instead of killing its process group or container.
