.. _using_docker_swarm:

Docker Swarm: Deployment Considerations
=======================================

This is a community experience report and architecture discussion, not a ready-to-run deployment guide. The contributor tested Xinference 2.2.0 with Docker 29.2.1, Ubuntu 22.04 and NVIDIA driver 570 on three machines, each with one NVIDIA L20. Xinference 3.x has not been validated in this Swarm setup.

Why consider Swarm?
-------------------

For a small team managing a dozen or so similar GPU servers, Swarm can reduce the manual work of deploying, restarting and updating containers across machines. It is built into Docker Engine. This was the contributor's reason for choosing it, not a rule that small clusters should always prefer Swarm.

Consider existing operational expertise, GPU management and availability requirements as well as cluster size. Teams already operating Kubernetes or needing its device-plugin ecosystem may prefer :ref:`using_kubernetes`.

Two layers of scheduling
------------------------

Swarm managers schedule container tasks on Docker nodes and maintain the services' desired state. Managers can also run tasks. A Xinference supervisor manages model placement and requests; Xinference worker processes register with it and run models. The supervisor does not provision worker containers: container orchestration and model serving are separate responsibilities.

A global Swarm service runs one task per eligible node, not one per GPU. GPU reservations and placement constraints must match the intended worker layout. Internal actor addresses must identify reachable individual processes; a load-balanced service address is not a substitute for each worker's address.

GPU management
--------------

Swarm generic resources are manually advertised scheduling resources, not automatic GPU discovery or health monitoring. With the resource kind ``NVIDIA-GPU``, Swarm exports allocated UUIDs through ``DOCKER_RESOURCE_NVIDIA-GPU``. NVIDIA runtime configuration must read that same variable: ``swarm-resource = "DOCKER_RESOURCE_NVIDIA-GPU"``. Scheduling reservations alone do not enforce GPU visibility.

Operators must keep advertised GPU UUIDs current and handle failed GPUs. This manual work was acceptable for the contributor's stable, homogeneous machines, but it is an important trade-off when hardware changes frequently. See the `NVIDIA runtime implementation <https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/cmd/nvidia-container-runtime-hook/hook_config.go>`_ for how the configured variable is selected.

Persistence and recovery
------------------------

In current Xinference, disabling authentication does not make the supervisor stateless. Its default ``XINFERENCE_HOME`` contains launch history (including autostart configuration), monitoring configuration and system settings. Persist this state; persist authentication data and keys too when authentication is enabled. Environment variables for download settings do not replace these databases.

A host bind mount stays on that host. Use placement on the specific storage host, or suitable durable storage available after rescheduling. A manager-role constraint alone does not select a unique host in a multi-manager cluster. Persist worker caches and engine environments separately.

Swarm can replace failed containers, but that does not establish Xinference model recovery or uninterrupted service. Test worker reconnection and model recovery for the chosen release. A single-manager Swarm cannot reschedule tasks while that manager is unavailable; surviving tasks can continue running. See `Swarm administration <https://docs.docker.com/engine/swarm/admin_guide/>`_.

Further reading
---------------

See `Swarm concepts <https://docs.docker.com/engine/swarm/key-concepts/>`_ for the orchestration model, :ref:`using_docker_compose` for the existing Compose setup, and :doc:`/getting_started/migration_3_0` for Xinference 3.x changes. A complete Swarm configuration and current-version cluster validation are outside the scope of this report.
