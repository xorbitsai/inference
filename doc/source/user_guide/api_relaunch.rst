API hot replacement
===================

On Linux and macOS, ``xinference-supervisor`` can replace its API process while preserving the Supervisor, Workers and model processes. The public address stays the same. New connections enter the replacement; existing requests and streams finish on the previous API.

Send ``POST /v1/api/relaunch`` or run ``xinference relaunch-api``. These operations require an administrator session when authentication is enabled. The optional ``python_executable`` selects an absolute Python path on the Supervisor machine; that environment must already contain a compatible Xinference installation. Omit it to reuse the currently serving API's environment.

.. code-block:: bash

    xinference relaunch-api --endpoint http://127.0.0.1:9997 \
      --python-executable /opt/xinference-new/bin/python --request-id api-upgrade-1

The HTTP 202 response contains an ``operation_id``. Poll ``GET /v1/api-relaunches/{operation_id}`` or list operations with ``GET /v1/api-relaunches``. States are ``starting``, ``switching``, ``draining``, ``succeeded`` and ``failed``. ``draining`` means the new API is serving while the previous API finishes admitted requests.

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_api(
        python_executable="/opt/xinference-new/bin/python", request_id="api-upgrade-1"
    )
    status = client.get_api_relaunch(operation["operation_id"])

A failed startup leaves the previous API serving. Reuse ``request_id`` when retrying the same request. Only one replacement may run at a time. Operation history belongs to the stable service host and lasts until it stops. Local mode, Windows, Supervisor upgrades and database migrations are outside this operation.
