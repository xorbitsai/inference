.. _worker_hot_replacement:

Worker hot replacement
======================

Replace a distributed Worker on Linux or macOS while retaining its model processes. The Supervisor and API server keep running. Both Workers and the model processes must support the handoff protocol and use Xoscar 0.12.1 or later, compatible Python versions, the same Supervisor, GPU visibility and extensions. Local mode, models spanning Workers and Xavier or P/D deployments are currently unsupported.

For an existing Worker at ``127.0.0.1:9978``, start its replacement with a different actor port and a free metrics exporter port:

.. code-block:: bash

    xinference-worker -e http://127.0.0.1:9997 --worker-port 9998 \
      --replace-worker 127.0.0.1:9978 --metrics-exporter-port 19002

Model management changes pause during handoff; inference calls continue. Failure before ownership transfer restores the old Worker. After transfer starts, keep both processes alive while the replacement retries. The old Worker exits automatically after successful replacement.

The model processes retain their original Xinference and engine code and loaded weights. This replaces the Worker control process. Deploy both processes on the same machine, and configure the service manager to let retained processes outlive the old Worker instead of killing its process group or container.
