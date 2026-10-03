.. _using_docker_swarm:

========================================
Distributed Deployment with Docker Swarm
========================================

This document covers three topics:

1. Why Docker Swarm is a better choice than Kubernetes for a cluster of a dozen or so machines
2. Understanding Swarm by comparing it with Xinference's supervisor/worker
3. How to adapt the official compose file for Swarm, with an introduction to basic Swarm concepts and commands

Tested with Xinference 2.2.0, Docker 29.2.1, Ubuntu 22.04, and NVIDIA driver 570 on 3 machines, each with one NVIDIA L20. The adjustments for Xinference 3.x in Section 4.8 are based on the official documentation and source code and have not yet been verified in a deployment.

.. note::

   Both Swarm and Xinference have a role called "worker", but the term refers to different things. In this document, "Swarm node" means a physical machine that has joined the Swarm cluster, and "Xinference worker" means the Xinference process that runs inference.

1. Why Swarm is a better choice than Kubernetes for a cluster of a dozen or so machines
=================================================

Our setup is a dozen or so GPU servers, maintained by a small team, mainly running Xinference. What we need from an orchestration tool:

-  Deploy containers across multiple machines
-  Restart containers automatically when they exit abnormally
-  Let containers on different machines reach each other
-  Assign Xinference workers according to the number of GPUs
-  Upgrade versions across all machines in one step

Swarm handles all of these well, although GPU allocation requires extra manual configuration (see Section 1.1).

Advantages of Swarm:

-  It is built into Docker, so there is nothing extra to install. ``docker swarm init`` creates a cluster and ``docker swarm join`` adds a machine
-  The deployment file is a docker-compose file plus a ``deploy`` section, so anyone who can write compose files has little new to learn
-  Deployment is a single command: ``docker stack deploy -c docker-compose-swarm.yml xinference``

What Kubernetes adds on top, such as load-based autoscaling, multi-tenant access control, and the Helm and Operator ecosystem, is not needed at this scale. In exchange, it brings learning and maintenance costs: etcd, the API server, the scheduler, network plugins, and GPU device plugins each need someone who understands them, and each of them can fail. For a small team, Kubernetes is overkill.

Swarm does have limitations to be aware of:

-  **GPU support is Swarm's main weakness.** All of our machines have GPUs, so this directly affects deployment and operations. It is covered separately in Section 1.1
-  We use a single manager. While the manager machine is down, services cannot be deployed, upgraded, or adjusted; containers that are already running are not affected
-  There are fewer community resources than for Kubernetes, so less help is available when problems come up

If you later need to share GPUs among multiple teams or autoscale based on load, or if your organization already runs a Kubernetes cluster, Xinference also documents a Kubernetes deployment (see :ref:`using_kubernetes`), and you can migrate then.

1.1 GPU Support: What to Know Before Choosing Swarm
---------------------------------------------------

**Swarm does not know about GPUs**

Kubernetes itself does not know about GPUs either, but the agent Kubernetes runs on each node (kubelet) defines a gRPC interface called the device plugin interface. Any vendor can write a program that implements this interface to report which devices a node has and whether each one is healthy, and to decide which device a container gets when it starts. NVIDIA implements this interface in the NVIDIA device plugin. Once deployed on each node, the plugin detects the node's GPUs automatically, reports them, and marks a GPU as unavailable when it fails. The plugin itself has to be installed, and each node still needs the driver and the NVIDIA Container Toolkit.

Swarm has no such interface. It only has a general-purpose counting mechanism called generic resources:

1. When Docker starts, it reads ``node-generic-resources`` from ``/etc/docker/daemon.json``, written there by hand, and reports the values to the manager. The values do not change after that. Swarm does not know what these values represent; it treats them as a set of strings that can be allocated
2. At deployment time, Swarm picks one of the values for a container and passes it in through the environment variable ``DOCKER_RESOURCE_NVIDIA-GPU=<UUID>``
3. For the container to actually see only that GPU, nvidia-container-runtime has to read this environment variable. This requires setting ``swarm-resource = "DOCKER_RESOURCE_GPU"`` in ``/etc/nvidia-container-runtime/config.toml``

Swarm and NVIDIA work together through an agreed-upon environment variable name, not through a formal plugin mechanism.

**Comparison**

.. list-table::
   :header-rows: 1

   * -
     - Kubernetes + NVIDIA device plugin
     - Swarm
   * - GPU discovery
     - Detected automatically by the plugin
     - Written by hand into ``daemon.json`` on each machine
   * - Configuration needed
     - Install the plugin (or have the NVIDIA GPU Operator install it)
     - Edit ``daemon.json`` and ``config.toml`` on each machine, then restart Docker
   * - GPU failure
     - The plugin marks the GPU unhealthy and stops assigning it
     - Swarm is unaware and keeps assigning containers to it
   * - Replacing a GPU
     - Updated automatically by the plugin
     - The UUID changes; reconfigure and restart Docker
   * - Identification
     - Handled by the plugin
     - GPU UUIDs must be entered by hand; GPU indices do not work

**What this means in practice**

-  Every machine needs two extra configuration steps before deployment. If they are skipped or done wrong, the Xinference worker stays Pending with ``insufficient resources``. We ran into this during deployment: declaring the GPU by index (``NVIDIA-GPU=0``) caused this error, removing ``node-generic-resources`` caused the same error, and it only worked after switching to GPU UUIDs
-  These two steps must not be skipped when adding a machine; otherwise the worker does not start after ``docker swarm join``
-  After a GPU is replaced, the configuration must be redone
-  Swarm does not notice when a GPU fails. You have to detect the problem yourself through ``nvidia-smi`` or the Xinference Web UI, then take that machine out of the cluster (see "GPU failure" in the deployment guide)
-  The number of GPUs per machine determines how Xinference workers are laid out. With one GPU per machine, ``mode: global`` is the simplest; machines with multiple GPUs need a different approach (see "Multi-GPU machines" in the deployment guide)

We wrote a script, ``setup-gpu-swarm.sh``, that gets the GPU UUIDs and edits both configuration files automatically. It reduces manual work, but it still has to be run once on every machine, and again after a GPU is replaced.

**Why we still chose Swarm**

Our cluster has a dozen or so machines with identical configurations, and GPUs are rarely replaced. The manual configuration is a one-time task, and with the script it takes a few minutes per machine. Compared with deploying and maintaining a Kubernetes cluster, this cost is acceptable.

If the GPU models or counts in your cluster change often, or you need GPU failures to be handled automatically, Kubernetes has a clear advantage here.

2. Understanding Swarm Through Xinference's Supervisor/Worker
=============================================================

2.1 Similar Structure, Different Responsibilities
-------------------------------------------------

Both Swarm and Xinference follow a "one control node + multiple execution nodes" structure. If you are familiar with Xinference, you can use its supervisor/worker to understand Swarm's manager/node:

.. list-table::
   :header-rows: 1

   * -
     - Swarm
     - Xinference
   * - Control role
     - manager
     - supervisor
   * - Execution role
     - Swarm node (physical machine)
     - Xinference worker (process in a container)
   * - What it schedules
     - Containers
     - Models
   * - Resources it tracks
     - CPU, memory, and GPUs of each machine
     - GPU memory and model usage

Swarm decides which machine a container runs on, without knowing what runs inside the container. Xinference decides which worker loads a model and where inference requests go, but it does not start worker processes. In our deployment, these two layers work on top of each other.

The two layers meet at the GPU: Swarm assigns a GPU to a Xinference worker container, and the container can see only that GPU; the Xinference worker detects that GPU at startup and loads models on it.

2.2 The Key Difference: The Supervisor Is Passive, the Manager Is Active
------------------------------------------------------------------------

The similar structure raises a common question: if Xinference already has supervisor/worker, why use Swarm?

The answer lies in how the two work.

**Xinference's supervisor is passive.** It only accepts registrations: the worker process on each machine connects to the supervisor after starting, and only then does the supervisor know about that worker. The supervisor cannot start a worker on another machine. When a worker process exits, the supervisor can only remove it from its list; it cannot bring it back up.

**Swarm's manager is active.** It makes sure these processes exist: which machine to start each container on, with which image and arguments, and restarting containers when they fail.

So without Swarm, Xinference's supervisor/worker still works, but someone has to do the following by hand:

-  Log in to each machine and run ``docker run`` with the correct supervisor address, local IP, port, and other arguments
-  When upgrading Xinference, stop the container, pull the image, and restart it on each machine in turn
-  When adding a machine, deploy a worker on it manually
-  Log in to each machine to check status and logs
-  When the machine running the supervisor goes down, restart the supervisor on another machine manually

With Swarm:

-  Run ``docker stack deploy`` once on the manager, and the supervisor and workers on all machines are deployed together
-  When a container exits abnormally, Swarm restarts it automatically
-  After a new machine runs ``docker swarm join``, a Xinference worker is deployed on it automatically
-  To upgrade, change the image version in the compose file and run ``docker stack deploy`` again
-  Use ``docker service ps`` and ``docker service logs`` on the manager to check status and logs across all machines

In short: Xinference manages models, and Swarm manages containers.

2.3 What Swarm Does Not Solve
-----------------------------

-  When the machine running the supervisor goes down, Swarm restarts the supervisor on another machine, and Xinference workers can still find it through the service name. However, the new supervisor does not keep records of previously loaded models, and we have not verified whether Xinference workers re-register automatically
-  If the machine that goes down happens to be the manager, Swarm itself cannot reschedule anything

With only two or three machines, manual maintenance is not costly. The more machines there are, the more manual work Swarm saves.

3. Swarm Basics
===============

3.1 Node Roles
--------------

A Swarm cluster has two kinds of machines:

-  **manager**: the machine that runs ``docker swarm init``. It handles scheduling and management. All commands for deploying and inspecting services are run on the manager
-  **Regular node**: a machine that joins with ``docker swarm join``. It runs containers

The manager also runs containers, just like regular nodes.

Swarm supports multiple managers that back each other up (3 managers can tolerate the failure of 1); we use only one for simplicity. The token used to join decides the role: ``docker swarm join-token worker`` generates the token for a regular node, and ``docker swarm join-token manager`` generates the token for a manager.

3.2 Key Concepts
----------------

.. list-table::
   :header-rows: 1

   * - Concept
     - Description
   * - Stack
     - A group of services deployed from one compose file with ``docker stack deploy``. Our stack is named ``xinference``
   * - Service
     - Each service in the compose file, such as ``xinference-supervisor``. After deployment, the stack name is added as a prefix: ``xinference_xinference-supervisor``
   * - Task
     - A running instance of a service. Normally each replica is one task. When a container fails and restarts, the old task is marked Failed and Swarm creates a new task to replace it. Each line in the output of ``docker service ps`` is a task
   * - Container
     - The container that a task actually runs on a machine
   * - Overlay network
     - A virtual network that spans machines; containers on different machines reach each other by service name
   * - Ingress
     - An overlay network that Swarm creates automatically to forward published ports to containers

Hierarchy: Stack → Service → Task → Container

The same replica slot may have several task records (the current one and earlier failed ones), but at any moment only one task is running, corresponding to one running container.

You do not need to create or manage tasks, but you should be able to read the output of ``docker service ps`` when troubleshooting:

.. code-block:: text

   ID             NAME                              NODE        DESIRED STATE   CURRENT STATE    ERROR
   rdr5avu6kz8o   xinference_xinference-worker.xx   node-1      Running         Running          
   v3lu6wcwdtuj    \_ xinference_xinference-worker.xx node-1    Shutdown        Failed           "task: non-zero exit (1)"

Lines with ``\_`` are earlier tasks in the same slot, and the ``NODE`` column shows the machine the container is on.

3.3 Basic Commands
------------------

**Cluster management**

.. code-block:: bash

   # On the manager: create the cluster
   docker swarm init --advertise-addr <manager private IP>

   # On the manager: show the command for joining the cluster
   docker swarm join-token worker

   # On each other machine: join the cluster (copy the command from the previous step)
   docker swarm join --token SWMTKN-1-xxxxx <manager private IP>:2377

   # On the manager: list all nodes
   docker node ls

   # On a machine: leave the cluster
   docker swarm leave

   # On the manager: remove the record of a node that has left
   docker node rm <node ID or hostname>

``--advertise-addr`` tells other machines which IP to use to reach the manager. It can be omitted if the machine has only one network interface; with multiple network interfaces it is required, otherwise an error is reported.

**Deploying and inspecting services** (run on the manager)

.. code-block:: bash

   # Deploy or update (run again after editing the compose file)
   docker stack deploy -c docker-compose-swarm.yml xinference

   # List all services and their replica counts
   docker service ls

   # Show the tasks of a service: which machine and what state
   docker service ps xinference_xinference-worker

   # Show full error messages (truncated by default)
   docker service ps --no-trunc xinference_xinference-worker

   # Show logs, aggregated from the containers on all machines
   docker service logs xinference_xinference-worker
   docker service logs -f xinference_xinference-worker   # follow

   # Remove the whole stack
   docker stack rm xinference

**Entering a container**

.. code-block:: bash

   # First, on the manager, use docker service ps to find the machine (NODE column)
   # Then log in to that machine and run
   docker ps
   docker exec -it <container ID> bash

``docker stack``, ``docker service``, and ``docker node`` commands can only be run on the manager; ``docker ps`` and ``docker exec`` must be run on the machine where the container is running.

3.4 Compose Files Under Swarm
-----------------------------

The same compose file behaves differently when deployed with ``docker compose up`` and with ``docker stack deploy``:

-  ``depends_on`` is ignored. Swarm does not guarantee the start order of services, so applications have to retry on their own
-  ``restart: always`` has no effect; use ``deploy.restart_policy`` instead
-  ``build`` has no effect. Only prebuilt images can be used, and every machine must be able to pull them
-  The ``deploy`` section only takes full effect under Swarm: replica count (``replicas``), one per machine (``mode: global``), resource reservations, placement constraints, and so on are all set here
-  Configuration files are distributed with ``configs``. Swarm sends the file contents to the machine where the container runs, so you do not need a copy on every machine

4. Adapting the Official Compose File
=====================================

4.1 Starting Point
------------------

The starting point is `docker-compose-distributed.yml <https://github.com/xorbitsai/inference/blob/main/xinference/deploy/docker/docker-compose-distributed.yml>`__ in the Xinference repository.

This file defines one supervisor and two workers. ``docker compose`` runs on a single machine only, so the file actually demonstrates the distributed structure on one machine. The comments in the file also say it is an example with two workers, and more can be added by incrementing the numbering.

The repository also has a ``docker-compose.yml``. It is for standalone deployment: there is only one container, and the supervisor and worker run in the same process (``xinference-local``). It is not a suitable basis for a distributed deployment.

When Xinference 3.0 was released, the standalone ``docker-compose.yml`` was updated, but the distributed file was not (its health check still uses curl and its image is still ``latest``). So the adjustments needed for 3.x are based on the standalone file and the 3.0 migration guide.

4.2 The Supervisor Does Not Need a GPU
--------------------------------------

The original file uses a YAML anchor that makes the supervisor inherit the GPU configuration:

.. code-block:: yaml

   xinference: &xinference
     deploy:
       resources:
         reservations:
           devices:
             - capabilities: [gpu]

   xinference-supervisor:
     <<: *xinference    # inherits the GPU configuration

The supervisor only schedules; it does not run inference. The source code confirms this: the supervisor's startup code creates only a scheduling actor and an HTTP service, with no GPU-related code, while the worker's startup code detects the number of GPUs and passes the GPU list to the worker. This is a side effect of reusing the template. After the change, the supervisor does not request a GPU.

4.3 Merging the Workers into One Service
----------------------------------------

The original file defines each worker as a separate service (``xinference-worker-1``, ``xinference-worker-2``), each with its own port. After the change, there is a single ``xinference-worker`` service with ``mode: global``, so Swarm runs one on every machine. When a new machine joins the cluster, a worker is deployed on it automatically, without editing the compose file.

``mode: global`` suits machines with one GPU each. If machines have multiple GPUs, the worker layout needs to be reconsidered; see "Multi-GPU machines" in the deployment guide.

4.4 Changing ``--host`` to ``$(hostname -i)``
---------------------------------------------

This is the most important change. First, look at how Xinference registration works.

When a worker starts, it sends its own address (``--host`` plus ``--worker-port``) to the supervisor:

.. code-block:: python

   # worker side
   await self._supervisor_ref.add_worker(self.address)

When the supervisor receives it, it uses this address to connect back to the worker:

.. code-block:: python

   # supervisor side
   async def add_worker(self, worker_address: str):
       worker_ref = await xo.actor_ref(address=worker_address, uid=WorkerActor.default_uid())

The same applies in the other direction: at startup, the worker first asks the supervisor for its internal address through the REST API, then connects to the supervisor using that address.

So the value of ``--host`` **must be an address the other side can connect back to**.

**Why the original file can use hostnames on a single machine**

.. code-block:: bash

   xinference-worker -H xinference-worker-1 --worker-port 30001
   xinference-worker -H xinference-worker-2 --worker-port 30002

In a single-machine compose deployment, each service is defined separately and the service name is the hostname, which Docker DNS resolves to a unique container IP. All containers are on the same network and can reach each other.

**Neither condition holds under Swarm**

Once the workers become one service, all replicas share the service name ``xinference-worker`` and have no hostnames of their own. Swarm DNS offers only two kinds of resolution: ``xinference-worker`` resolves to a virtual IP that Swarm load-balances to one of the replicas, and ``tasks.xinference-worker`` returns the IPs of all replicas. Neither can point to a specific replica.

If ``-H 0.0.0.0`` is used instead, every worker registers the address ``0.0.0.0:30001``, and when the supervisor connects back to that address, it connects to itself.

The supervisor has the same problem. With ``--host 0.0.0.0``, the internal address the supervisor gives to workers is ``0.0.0.0:9999``, and a worker connecting to it connects to itself. This is the log from an actual deployment:

.. code-block:: text

   xinference.core.worker ERROR  Failed to upload node info: [Errno 111] Connect call failed ('0.0.0.0', 9999)

At that point, the cluster information page of the Web UI showed 0 workers.

**Solution**

Inside the container, use ``hostname -i`` to get the container's IP on the overlay network, and start with that IP:

.. code-block:: yaml

   command: sh -c 'xinference-supervisor --host $$(hostname -i) --port 9997 --supervisor-port 9999'
   command: sh -c 'xinference-worker -e http://xinference-supervisor:9997 --host $$(hostname -i) --worker-port 30001'

Each container has a unique IP on the overlay network that the others can reach, with no dependence on DNS. ``$$`` is an escape in compose files; the shell receives ``$(hostname -i)``.

After the change, the supervisor address is an overlay IP such as ``10.0.1.3:9999``, and the 3 workers register as ``10.0.1.6:30001``, ``10.0.1.7:30001``, and ``10.0.1.8:30001``.

4.5 Why nginx Is Needed
-----------------------

After ``--host`` is changed to the overlay IP, workers can register, but the Web UI can no longer be reached from outside. The reason is that ``--host`` controls two things at once:

-  **The REST API listening address** (port 9997): the Web UI and HTTP API, which must be reachable from outside
-  **The actor address** (port 9999): internal communication between the supervisor and workers, which must give workers an address they can connect back to

.. code-block:: python

   # supervisor startup code
   supervisor_address = f"{host}:{supervisor_port}"    # actor address
   restful_api.run(supervisor_address=supervisor_address, host=host, port=port)   # the REST API uses the same host

The two have conflicting requirements for the address:

-  ``--host 0.0.0.0``: the REST API listens on all network interfaces and is reachable from outside, but the actor address becomes ``0.0.0.0:9999`` and worker registration fails
-  ``--host <overlay IP>``: worker registration works, but the REST API listens only on the overlay network

Why does listening only on the overlay network make it unreachable from outside? In Swarm, a container that publishes a port has two network interfaces: one on the ingress network and one on the overlay network we defined (``xinference-net``). External traffic to ``<machine IP>:9997`` enters the container through the ingress network, but the REST API listens only on the overlay IP, so it never receives that traffic.

A single-machine compose deployment does not have this problem: the container has only one network interface, so the port forwarding target and the address the service listens on are the same IP.

Xinference's command line has no separate option for the REST API listening address (as of 3.5.0, the supervisor's only options are ``--host``, ``--port``, ``--supervisor-port``, and ``--log-level``), so an nginx container is added to forward traffic:

-  The supervisor does not publish a port and starts with its overlay IP, so internal communication works
-  nginx publishes port 9997, listens on all network interfaces, and receives external traffic
-  nginx forwards requests over the overlay network to the supervisor, using the service name ``xinference-supervisor``

.. code-block:: text

   Browser → any machine IP:9997 → ingress network → nginx container → overlay network → supervisor (10.0.1.x:9997)

Ports published by Swarm are reachable on every machine in the cluster, so the Web UI can be opened through port 9997 on any machine's IP.

If Xinference later provides separate options for the REST API listening address and the actor address, nginx can be removed.

4.6 Requesting GPUs
-------------------

The background for this change is in Section 1.1: Swarm does not know about GPUs, so GPUs can only be declared and allocated by hand through generic resources.

On a single Docker host, GPUs can be used directly once the NVIDIA Container Toolkit is installed, and the original file uses this single-host syntax:

.. code-block:: yaml

   # Original file: single-host docker compose syntax
   deploy:
     resources:
       reservations:
         devices:
           - capabilities: [gpu]
             driver: nvidia
             count: all

A single host involves no scheduling; Docker mounts the local GPUs into the container directly. Swarm has to decide which machine each container goes on, so it needs to know how many GPUs each machine has and how many are already assigned. That is why the syntax changes to generic resources:

.. code-block:: yaml

   # After the change: Swarm syntax
   deploy:
     resources:
       reservations:
         generic_resources:
           - discrete_resource_spec:
               kind: 'NVIDIA-GPU'
               value: 1      # each worker requests 1 GPU

``kind`` must match the name used in ``node-generic-resources`` in each machine's ``daemon.json`` (``NVIDIA-GPU``). When scheduling, Swarm matches the number of GPUs declared on each machine against the number each worker requests; machines with no GPUs left are not assigned a worker.

The compose file is only half of this mechanism. The other half is in ``daemon.json`` and ``config.toml`` on each machine, and both halves must be configured for it to work.

4.7 Start Order
---------------

Swarm ignores ``depends_on``, so the supervisor and workers start at almost the same time. If a worker starts before the supervisor is ready, the connection fails and the container exits.

This is handled by the restart policy: the worker's ``restart_policy.delay`` is set to 20 seconds, so after a failure it waits 20 seconds before restarting, by which time the supervisor is up. In our tests, with 10 seconds each worker failed four times before succeeding; with 20 seconds it succeeded on the first try.

4.8 Adjustments for Xinference 3.x
----------------------------------

Several changes in Xinference 3.0 affect deployment. The following adjustments were made on 3.5.0:

.. list-table::
   :header-rows: 1

   * - Change
     - Adjustment
   * - The official GPU image is based on CUDA 13 and requires NVIDIA driver 580 or later
     - Upgrade the driver on all machines
   * - Authentication is enabled by default; every API call requires a login or an API key
     - For use on an internal network, disable it with ``XINFERENCE_AUTH_ADVANCED=false``
   * - The image no longer bundles inference engines such as vLLM and SGLang; they are installed into ``XINFERENCE_HOME/virtualenv`` the first time a model is launched
     - Mount the worker's ``/root/.xinference`` to a host directory so engines are not reinstalled when the container is recreated; configure a pip mirror reachable from your network
   * - The official docs state that the image is guaranteed to contain python3, but not curl
     - Use python3 for the health check
   * - The Web UI moved from ``/ui`` to ``/``
     - Open the Web UI at ``http://<IP>:9997/``
   * - The official recommendation is to pin the image version in production
     - Use ``xprobe/xinference:v3.5.0`` instead of ``latest``, so machines do not pull different versions at different times

With authentication disabled, the only data the supervisor needs to persist is ``XINFERENCE_HOME/system-settings.json``, which holds model download settings (download source, pip index, and so on). If these settings are changed in the Web UI, they are lost when the supervisor is rescheduled to another machine. So keep these settings in environment variables in the compose file instead of changing them in the Web UI. That way, the supervisor needs no mounted directory and does not have to be pinned to a particular machine.

If you later enable authentication, the authentication database and encryption key are stored under the supervisor's ``XINFERENCE_HOME/auth``. In that case, pin the supervisor to the manager (``placement.constraints: [node.role == manager]``, a built-in Swarm attribute that requires no labels) and mount ``/root/.xinference`` to a host directory.
