.. _zero_downtime_upgrade:

Zero-downtime upgrades (Experimental)
=====================================

.. versionadded:: v4.1.0

Upgrade one component at a time while inference continues. Choose the component whose code or environment needs changing; replacing a control process keeps existing model processes and their loaded weights running.

Prepare an upgrade
------------------

The running deployment must already support the relevant replacement protocol and service host. API, Supervisor and Worker replacement support distributed deployments on Linux and macOS, not local mode or Windows. Model-engine replacement has the Linux GPU requirements described below.

Install a compatible Xinference version and its dependencies in a separate environment first. ``python_executable`` is an absolute Python path on the Supervisor machine for API/Supervisor upgrades, or on the Worker machine for Worker/model upgrades. Packages are not installed by these operations. Omit the path to reuse the current environment. Use an administrator session when authentication is enabled.

API and Web UI
--------------

In **Cluster Information**, choose **Restart / upgrade API**, or run:

.. code-block:: bash

    xinference relaunch-api --endpoint http://127.0.0.1:9997 \
      --python-executable /opt/xinference-new/bin/python --request-id api-upgrade-1

The public HTTP address stays the same. New connections use the replacement API; existing requests and streams finish on the previous API. The Supervisor, Workers and models keep running. Refresh the browser to load the upgraded Web UI. See :doc:`api_relaunch`.

Supervisor
----------

In **Cluster Information**, choose **Restart / upgrade Supervisor**, or run:

.. code-block:: bash

    xinference relaunch-supervisor --endpoint http://127.0.0.1:9997 \
      --python-executable /opt/xinference-new/bin/python --request-id supervisor-upgrade-1

The API, Workers and models keep running, and public addresses stay the same. Inference routes remain available while management operations briefly wait during state transfer. Finish active model launches, downloads, reloads and Worker handoffs before upgrading. See :doc:`supervisor_relaunch`.

Worker
------

In **Cluster Information**, choose **Restart / upgrade Worker** on the Worker to upgrade, or use its current actor address:

.. code-block:: python

    from xinference.client import Client

    client = Client("http://127.0.0.1:9997")
    operation = client.relaunch_worker(
        "127.0.0.1:9978",
        python_executable="/opt/xinference-new/bin/python",
        request_id="worker-upgrade-1",
    )
    status = client.get_worker_relaunch(operation["operation_id"])

The Worker starts its replacement and transfers model ownership automatically. Model processes, their code and GPU weights stay in place; this does not upgrade the model engine. Both environments need Xoscar 0.12.1 or later and compatible handoff support. Multi-Worker models and Xavier/P/D deployments are unsupported. See :doc:`worker_handoff`.

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

Progress and recovery
---------------------

Submission returns HTTP 202 and an ``operation_id``; acceptance does not mean completion. Follow progress in the Web UI or the component's status API. API, Supervisor and Worker operations finish with ``succeeded`` or ``failed``; model reload finishes with ``ready`` or ``error``. API ``draining`` means the replacement is serving while previous requests finish.

Reuse the same ``request_id`` when retrying an API, Supervisor or Worker submission. If a Worker operation is ``pending``, keep both processes alive and continue polling: ownership transfer is uncertain. Failed preparation keeps the previous component serving; for model errors, ``restored=true`` confirms the previous engine is serving.

Upgrade boundaries
------------------

Keep service hosts and containers running throughout an upgrade. Stopping them also stops retained processes and releases GPU allocations. Supervisor support actors retain their original environments. Updating these hosts, retained support actors or database schemas requires a planned restart; these operations do not upgrade an entire installation at once.

Component reference
-------------------

.. toctree::
   :maxdepth: 1

   api_relaunch
   supervisor_relaunch
   worker_handoff
