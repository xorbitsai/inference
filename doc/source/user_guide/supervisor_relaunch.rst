Supervisor hot replacement
==========================

On Linux and macOS, ``xinference-supervisor`` can replace its Supervisor control process while retaining the API, Workers and model processes. The HTTP address and Supervisor actor address stay the same. New inference requests continue using the serving routes while the replacement prepares; existing streams stay on their model processes.

Use ``POST /v1/supervisor/relaunch``, ``xinference relaunch-supervisor`` or ``Client.relaunch_supervisor()``. Authentication requires an administrator session. The optional ``python_executable`` must be an absolute path to an already prepared compatible Xinference environment on the Supervisor machine. Omit it to reuse the serving Supervisor's environment.

.. code-block:: bash

    xinference relaunch-supervisor --endpoint http://127.0.0.1:9997 \
      --python-executable /opt/xinference-new/bin/python --request-id supervisor-upgrade-1

The replacement loads persisted configuration and checks the registered Workers' live model inventories and ownership. Management operations briefly wait during state transfer; model route lookups remain available. An active model launch, download, model reload or Worker handoff prevents replacement until that operation finishes. Failed preparation leaves the previous Supervisor serving.

The HTTP 202 response includes an ``operation_id``. Poll ``GET /v1/supervisor-relaunches/{operation_id}`` or list operations with ``GET /v1/supervisor-relaunches``. States are ``starting``, ``preparing``, ``switching``, ``draining``, ``succeeded`` and ``failed``. Only one Supervisor replacement runs at a time. Reuse ``request_id`` when retrying the same operation; history lasts until the stable service host stops.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_supervisor(request_id="supervisor-upgrade-1")
    status = client.get_supervisor_relaunch(operation["operation_id"])

The service must have been started with Supervisor hosting support. The stable host and its state, cache, event, progress and model-routing actors retain their original environments. This operation upgrades Supervisor control logic; upgrading retained actors or changing database schemas requires a planned restart. Local mode and Windows are unsupported.
