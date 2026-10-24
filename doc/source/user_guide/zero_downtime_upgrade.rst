.. _zero_downtime_upgrade:

Zero-downtime upgrades (Experimental)
=====================================

.. versionadded:: v4.1.0

Upgrade one component at a time while inference continues. Choose the component whose code or environment needs changing; replacing a control process keeps existing model processes and their loaded weights running.

Prepare an upgrade
------------------

The running deployment must already support the relevant replacement protocol and service host. API, Supervisor and Worker replacement support local and distributed deployments on Linux and macOS. Windows is unsupported. Model-engine replacement has the Linux GPU requirements described below.

Install a compatible Xinference version and its dependencies in a separate environment first. ``python_executable`` is an absolute Python path on the Supervisor machine for API/Supervisor upgrades, or on the Worker machine for Worker/model upgrades. Packages are not installed by these operations. Omit the path to reuse the current environment. Use an administrator session when authentication is enabled.

In local mode, startup still uses an API parent and one shared Supervisor/Worker process. A component moves into a separate process only on its first upgrade; use the same commands and Web UI actions below. The original parent and shared process stay alive as service hosts, so stopping the original local service stops its replacement processes too.

API and Web UI
--------------

In **Cluster Information**, choose **Restart / upgrade API**, or run:

.. code-block:: bash

    xinference relaunch-api --endpoint http://127.0.0.1:9997 \
      --python-executable /opt/xinference-new/bin/python --request-id api-upgrade-1

The public HTTP address stays the same. New connections use the replacement API; existing requests and streams finish on the previous API. The Supervisor, Workers and models keep running. Refresh the browser to load the upgraded Web UI.

REST API: ``POST /v1/api/relaunch``.

The HTTP 202 response contains an ``operation_id``. Poll ``GET /v1/api-relaunches/{operation_id}`` or list operations with ``GET /v1/api-relaunches``. States are ``starting``, ``switching``, ``draining``, ``succeeded`` and ``failed``. ``draining`` means the new API is serving while the previous API finishes admitted requests.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_api(
        python_executable="/opt/xinference-new/bin/python", request_id="api-upgrade-1"
    )
    status = client.get_api_relaunch(operation["operation_id"])

A failed startup leaves the previous API serving. Reuse ``request_id`` when retrying the same request. Only one replacement may run at a time. Operation history belongs to the stable service host and lasts until it stops. Windows, Supervisor upgrades and database migrations are outside this operation.

Supervisor
----------

In **Cluster Information**, choose **Restart / upgrade Supervisor**, or run:

.. code-block:: bash

    xinference relaunch-supervisor --endpoint http://127.0.0.1:9997 \
      --python-executable /opt/xinference-new/bin/python --request-id supervisor-upgrade-1

On Linux and macOS, ``xinference-supervisor`` and ``xinference-local`` can replace its Supervisor control process while retaining the API, Workers and model processes. The HTTP address and Supervisor actor address stay the same. New inference requests continue using the serving routes while the replacement prepares; existing streams stay on their model processes.

The replacement loads persisted configuration and checks the registered Workers' live model inventories and ownership. Management operations briefly wait during state transfer; model route lookups remain available. An active model launch, download, model reload or Worker handoff prevents replacement until that operation finishes. Failed preparation leaves the previous Supervisor serving.

REST API: ``POST /v1/supervisor/relaunch``.

The HTTP 202 response includes an ``operation_id``. Poll ``GET /v1/supervisor-relaunches/{operation_id}`` or list operations with ``GET /v1/supervisor-relaunches``. States are ``starting``, ``preparing``, ``switching``, ``draining``, ``succeeded`` and ``failed``. Only one Supervisor replacement runs at a time. Reuse ``request_id`` when retrying the same operation; history lasts until the stable service host stops.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_supervisor(request_id="supervisor-upgrade-1")
    status = client.get_supervisor_relaunch(operation["operation_id"])

The service must have been started with Supervisor hosting support. The stable host and its state, cache, event, progress and model-routing actors retain their original environments. This operation upgrades Supervisor control logic; upgrading retained actors or changing database schemas requires a planned restart. Windows is unsupported.

.. _worker_hot_replacement:

Worker
------

Replace a Worker on Linux or macOS while retaining its model processes. The Supervisor and API server keep running. Both Workers and the model processes must support the handoff protocol and use Xoscar 0.12.1 or later, compatible Python versions, the same Supervisor, GPU visibility and extensions. Models spanning Workers and Xavier or P/D deployments are currently unsupported.

In **Cluster Information**, administrators can use **Restart / upgrade Worker** on a Worker. Operation records remain visible when its address changes or the page is refreshed. This retains the current model processes; model-engine upgrades use **Reconfigure / upgrade model** in the running model's detail view.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_worker(
        "127.0.0.1:9978",
        python_executable="/opt/xinference-new/bin/python",
        request_id="worker-upgrade-1",
    )
    status = client.get_worker_relaunch(operation["operation_id"])

To relaunch an existing Worker, send ``POST /v1/workers/{worker_address}/relaunch``. The Worker starts its replacement automatically and inherits its Supervisor, GPU visibility, runtime configuration and configured metrics exporter port; a new actor port is selected automatically.

.. code-block:: bash

    curl -X POST 'http://127.0.0.1:9997/v1/workers/127.0.0.1:9978/relaunch' \
      -H 'Content-Type: application/json' -d '{"request_id":"upgrade-1"}'

The response is HTTP 202 with an ``operation_id``. Poll ``GET /v1/worker-relaunches/{operation_id}`` or list recent operations with ``GET /v1/worker-relaunches``. Status is one of ``starting``, ``transferring``, ``pending``, ``succeeded`` or ``failed``. These routes require administrator access when authentication is enabled.

The Supervisor stores operation records in ``XINFERENCE_HOME/worker-relaunches.json``, so status remains available after the old Worker exits and can be reconciled after a Supervisor restart. Duplicate requests reuse an active operation. Keep the same optional ``request_id`` when retrying a request; use a new ID for a new operation. ``pending`` means the transfer outcome is uncertain: keep the processes alive and continue polling instead of starting another replacement.

The Supervisor also stores ownership reservations in ``XINFERENCE_HOME/worker-handoffs.json`` so an interrupted handoff can resume after a Supervisor restart. Replacement Workers keep updating the recovery file for the original Worker address, including models launched after relaunch.

Model management changes pause during handoff; inference calls continue. Failure before ownership transfer restores the old Worker. After transfer starts, keep both processes alive while the replacement retries. The old Worker stops automatically after successful replacement.

The model processes retain their original Xinference and engine code and loaded weights. This replaces the Worker control process. Deploy both processes on the same machine, and configure the service manager to let retained processes outlive the old Worker instead of killing its process group or container.

Service and container ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Model and engine
----------------

Open a running model's detail view and choose **Reconfigure / upgrade model**. Engine replacement requires **Keep GPU weights for reload** (``enable_weight_cache=true``) at launch, vLLM 0.31.0+ or SGLang 0.5.21+, Linux with CUDA or ROCm, one Worker, one replica and fixed tensor parallelism. To select a prepared model environment:

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.reload_model(
        "my-model", {}, python_executable="/opt/xinference-new/bin/python"
    )
    status = client.get_model_reload_status("my-model")

The current engine serves until its replacement is healthy, then existing requests finish on the old engine. Compatible engine and PyTorch versions can share retained weights, but each engine needs its own KV cache and CUDA graphs. Reserve extra GPU memory; incompatible versions also need a separate weight cache. Insufficient memory rejects the upgrade while the current engine keeps serving. See :ref:`launch_weight_cache_reload` for supported changes and limitations.

Send ``POST /v1/models/{model_uid}/reload`` to submit a model upgrade. Poll ``GET /v1/models/{model_uid}/reload`` until ``status`` is ``ready`` or ``error``. On error, ``restored=true`` confirms that the previous engine is serving.

Upgrade boundaries
------------------

Keep service hosts and containers running throughout an upgrade. Stopping them also stops retained processes and releases GPU allocations. Supervisor support actors retain their original environments. Updating these hosts, retained support actors or database schemas requires a planned restart; these operations do not upgrade an entire installation at once.
