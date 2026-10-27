.. _zero_downtime_upgrade:

Zero-downtime upgrades (Experimental)
=====================================

.. versionadded:: v4.1.0

Upgrade one component at a time while inference continues. Choose the component whose code or environment needs changing; replacing a control process keeps existing model processes and their loaded weights running.

Prepare an upgrade
------------------

The running deployment must already support the relevant replacement protocol and service host. API, Supervisor and Worker replacement support local and distributed deployments on Linux, macOS and Windows. Model-engine replacement has the Linux GPU requirements described below. On Windows, run original and replacement processes under the same user and privilege level. On Windows, the local service host and API subprocesses use selector event loops, each limited to 512 sockets.

By default, a replacement uses the component's current Python environment. No Python path is required. Use an administrator session when authentication is enabled.

In local mode, startup still uses an API parent and one shared Supervisor/Worker process. A component moves into a separate process only on its first upgrade; use the same commands and Web UI actions below. The original parent and shared process stay alive as service hosts, so stopping the original local service stops its replacement processes too. If a migrated Worker exits unexpectedly, the API and Supervisor stay available; attach a Worker to restore serving capacity.

Check and install updates
-------------------------

Use the version control at the bottom left of the Web UI to check for updates. Choose **Install and upgrade** to update Workers, then Supervisor, then API. Each component shows its progress; the displayed version changes after the API takes over. Detection uses the configured package source (``PIP_INDEX_URL``, or PyPI by default) on each component's machine and offers newer stable versions compatible with its Python.

Xinference installs the selected version and its dependencies into a separate directory under ``XINFERENCE_HOME/upgrades``. Status shows ``installing`` and ``migrating`` before the usual replacement states. Preparation failure leaves the current component serving. Existing model environments and service hosts keep running; prepared environments are retained for diagnosis and reuse.

Worker upgrades preserve the source environment's package versions, including inference engines and the Torch build. The new Worker environment reuses base Python packages and reinstalls the source environment's own packages. If the selected release requires incompatible dependencies, preparation fails and the existing Worker keeps serving; use a manually prepared compatible environment for that upgrade.

API and Supervisor upgrades also preserve source packages when guard protection or ``XINFERENCE_EXTENSIONS`` is enabled. Guard-enabled upgrades install the target release's ``guard`` extra. Dependency conflicts fail preparation while the current component keeps serving.

For preserved PyTorch builds with CUDA, CPU, ROCm or XPU version tags, Xinference first checks the configured package source. If that source cannot provide the build, uv selects the matching official PyTorch backend; other packages keep the configured source. The original build remains pinned. Bundled ``pip`` is excluded from the runtime package profile.

Use ``GET /v1/api/updates``, ``GET /v1/supervisor/updates`` or ``GET /v1/workers/{worker_address}/updates`` to check a component. Pass ``version`` to its existing relaunch endpoint to install and upgrade; ``version`` and ``python_executable`` are mutually exclusive. For a private source or integration tests, set ``XINFERENCE_UPGRADE_INDEX_URL`` before starting the service on each relevant machine. The override applies to both detection and installation.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    update = client.check_component_update("api")
    if update["update_available"]:
        operation = client.relaunch_api(version=update["latest_version"])

A release can register optional ``xinference.migrations`` entry points in its package metadata. Each hook runs in the installed target environment before replacement and receives a dictionary with ``component``, ``source_version``, ``target_version`` and ``home``. Hooks must be safe to retry and compatible with components still serving. A hook failure stops the upgrade; the framework does not undo its data changes.

.. code-block:: toml

    [project.entry-points."xinference.migrations"]
    prepare = "xinference.deploy.migrations:migrate"

Test with a mock package source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following command from the checkout in the Xinference environment. It serves the current code as a higher version of Xinference and forwards dependency requests to ``PIP_INDEX_URL`` (or PyPI). Use ``--upstream`` to select a different dependency source. It does not start or replace a Xinference service.

.. code-block:: bash

    python scripts/mock_upgrade_index.py --version 9999.0.0 --port 18080

Set ``XINFERENCE_UPGRADE_INDEX_URL=http://127.0.0.1:18080/simple`` before starting your normal Xinference service. Open the version control at the bottom left, check for updates and install. Confirm that ``GET /v1/cluster/version`` reports ``9999.0.0`` and the service still handles requests. Run the mock source on the same machine as the service, or use an address reachable by every component.

.. code-block:: bash

    export XINFERENCE_UPGRADE_INDEX_URL=http://127.0.0.1:18080/simple
    xinference-local --host 127.0.0.1 --port 9997

API and Web UI
--------------

In **Cluster Information**, choose **Restart / upgrade API**, or run:

.. code-block:: bash

    xinference relaunch-api --endpoint http://127.0.0.1:9997

.. figure:: /_static/supervisor_restart.png
   :alt: Restart or upgrade the API and Supervisor from Cluster Information.
   :width: 100%
   :align: center

   Restart or upgrade the API and Supervisor from Cluster Information.

The public HTTP address stays the same. New connections use the replacement API; existing requests and streams finish on the previous API. The Supervisor, Workers and models keep running. Refresh the browser to load the upgraded Web UI.

REST API: ``POST /v1/api/relaunch``.

The HTTP 202 response contains an ``operation_id``. Poll ``GET /v1/api-relaunches/{operation_id}`` or list operations with ``GET /v1/api-relaunches``. States are ``installing``, ``migrating``, ``starting``, ``switching``, ``draining``, ``succeeded`` and ``failed``. ``draining`` means the new API is serving while the previous API finishes admitted requests.

A failed startup leaves the previous API serving. Reuse ``request_id`` when retrying the same request. Only one replacement may run at a time. Operation history belongs to the stable service host and lasts until it stops. Supervisor upgrades are separate operations.

Supervisor
----------

In **Cluster Information**, choose **Restart / upgrade Supervisor**, or run:

.. code-block:: bash

    xinference relaunch-supervisor --endpoint http://127.0.0.1:9997

On Linux, macOS and Windows, ``xinference-supervisor`` and ``xinference-local`` can replace their Supervisor control process while retaining the API, Workers and model processes. The HTTP address and Supervisor actor address stay the same. New inference requests continue using the serving routes while the replacement prepares; existing streams stay on their model processes.

The replacement loads persisted configuration and checks the registered Workers' live model inventories and ownership. Management operations briefly wait during state transfer; model route lookups remain available. An active model launch, download, model reload or Worker handoff prevents replacement until that operation finishes. Failed preparation leaves the previous Supervisor serving.

REST API: ``POST /v1/supervisor/relaunch``.

The HTTP 202 response includes an ``operation_id``. Poll ``GET /v1/supervisor-relaunches/{operation_id}`` or list operations with ``GET /v1/supervisor-relaunches``. States are ``installing``, ``migrating``, ``starting``, ``preparing``, ``switching``, ``draining``, ``succeeded`` and ``failed``. Only one Supervisor replacement runs at a time. Reuse ``request_id`` when retrying the same operation; history lasts until the stable service host stops.

The service must have been started with Supervisor hosting support. The stable host and its state, cache, event, progress and model-routing actors retain their original environments. This operation upgrades Supervisor control logic; upgrading retained actors requires a planned restart.

.. _worker_hot_replacement:

Worker
------

Replace a Worker on Linux, macOS or Windows while retaining its model processes. The Supervisor and API server keep running. Both Workers and the model processes must support the handoff protocol and use Xoscar 0.12.2 or later, compatible Python versions, the same Supervisor, GPU visibility and extensions. Models spanning Workers and Xavier or P/D deployments are currently unsupported.

Worker relaunch requires the target Xinference version to support the running host's registration protocol. A downgrade to a version without that support fails while the current Worker keeps serving.

In **Cluster Information**, administrators can use **Restart / upgrade Worker** on a Worker. Operation records remain visible when its address changes or the page is refreshed. This retains the current model processes; model-engine upgrades use **Reconfigure / upgrade model** from **Running Models** or the model's detail view.

.. figure:: /_static/worker_restart.png
   :alt: Restart or upgrade a Worker from its row in Cluster Information.
   :width: 100%
   :align: center

   Restart or upgrade a Worker from its row in Cluster Information.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_worker("127.0.0.1:9978")
    status = client.get_worker_relaunch(operation["operation_id"])

To relaunch an existing Worker, send ``POST /v1/workers/{worker_address}/relaunch``. The Worker starts its replacement automatically and inherits its Supervisor, GPU visibility, runtime configuration and configured metrics exporter port; a new actor port is selected automatically.

The response is HTTP 202 with an ``operation_id``. Poll ``GET /v1/worker-relaunches/{operation_id}`` or list recent operations with ``GET /v1/worker-relaunches``. Status is one of ``installing``, ``migrating``, ``starting``, ``transferring``, ``pending``, ``succeeded`` or ``failed``. These routes require administrator access when authentication is enabled.

The Supervisor stores operation records in ``XINFERENCE_HOME/worker-relaunches.json``, so status remains available after the old Worker exits and can be reconciled after a Supervisor restart. Duplicate requests reuse an active operation. Keep the same optional ``request_id`` when retrying a request; use a new ID for a new operation. ``pending`` means the transfer outcome is uncertain: keep the processes alive and continue polling instead of starting another replacement.

The Supervisor also stores ownership reservations in ``XINFERENCE_HOME/worker-handoffs.json`` so an interrupted handoff can resume after a Supervisor restart. Replacement Workers keep updating the recovery file for the original Worker address, including models launched after relaunch.

Model management changes pause during handoff; inference calls continue. Failure before ownership transfer restores the old Worker. After transfer starts, keep both processes alive while the replacement retries. The old Worker stops automatically after successful replacement.

The model processes retain their original Xinference and engine code and loaded weights. This replaces the Worker control process. Deploy both processes on the same machine, and configure the service manager to let retained processes outlive the old Worker instead of killing its process group or container.

Service and container ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Linux, macOS and Windows, ``xinference-worker`` automatically keeps a stable Worker host as its main process. Replacement Workers register before handoff, so retirement of the original Worker does not stop its models, replacement, service or container. Explicitly stopping the host terminates its owned processes. Custom launch commands can use the host wrapper below.

On Windows, stopping the Worker host forcibly terminates its owned processes. The Supervisor detects the stopped Worker after its heartbeat timeout.

.. code-block:: bash

    python -m xinference.deploy.worker_host xinference-worker \
      --endpoint http://127.0.0.1:9997 --host 0.0.0.0

A systemd service can keep its normal process-group cleanup, because the Worker host remains its main process.

For Docker, use the same host command as the container command. Install or mount prepared Python environments inside the running container. Keep that container running during upgrades: replacing or stopping the container terminates its model processes and cannot preserve their GPU allocations.

Model and engine
----------------

Open a running model's detail view and choose **Reconfigure / upgrade model**. Engine replacement requires **Keep GPU weights for reload** (``enable_weight_cache=true``) at launch, vLLM 0.31.0+ or SGLang 0.5.21+, Linux with CUDA or ROCm, one Worker, one replica and fixed tensor parallelism.

.. figure:: /_static/reconfigure_in_running.png
   :alt: Open model reconfiguration from Running Models.
   :width: 100%
   :align: center

   Open model reconfiguration from Running Models.

.. figure:: /_static/reconfigure_in_detail.png
   :alt: Open Reconfigure / upgrade model from the model's detail view.
   :width: 100%
   :align: center

   Open Reconfigure / upgrade model from the model's detail view.

For vLLM:

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.reload_model("my-model", {"max_num_seqs": 32})
    status = client.get_model_reload_status("my-model")

The current engine serves until its replacement is healthy, then existing requests finish on the old engine. Compatible engine and PyTorch versions can share retained weights, but each engine needs its own KV cache and CUDA graphs. Reserve extra GPU memory; incompatible versions also need a separate weight cache. Insufficient memory rejects the upgrade while the current engine keeps serving. See :ref:`launch_weight_cache_reload` for supported changes and limitations.

Send ``POST /v1/models/{model_uid}/reload`` to submit a model upgrade. Poll ``GET /v1/models/{model_uid}/reload`` until ``status`` is ``ready`` or ``error``. On error, ``restored=true`` confirms that the previous engine is serving.

Switch environments (optional)
------------------------------

You can also prepare a compatible Xinference environment manually. Pass its absolute Python path with ``python_executable`` in Python/REST calls or ``--python-executable`` in CLI commands. API/Supervisor paths are on the Supervisor machine; Worker/model paths are on the Worker machine. Omit ``version`` when using a prepared environment.

Upgrade boundaries
------------------

Keep service hosts and containers running throughout an upgrade. Stopping them also stops retained processes and releases GPU allocations. Supervisor support actors retain their original environments. Updating these hosts or retained support actors requires a planned restart; these operations do not upgrade an entire installation at once.
