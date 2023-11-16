.. _using_xinference:

================
Using Xinference
================

Configure Xinference Home Path
==============================
By default, Xinference uses ``<HOME>/.xinference`` as home path to store necessary files such as logs and models,
where ``<HOME>`` is the home path of current user.

You can change this directory by configuring the environment variable ``XINFERENCE_HOME``.
For example:

.. code-block:: bash

  export XINFERENCE_HOME=/tmp/xinference


Using Xinference Locally
========================

To start a local instance of Xinference, run the following command:

.. code-block:: bash

  xinference-local --host 0.0.0.0 --port 9997


Using Xinference In a Cluster
=============================


To deploy Xinference in a cluster, you need to start a Xinference supervisor on one server and Xinference workers
on the other servers. Follow the steps below:

Starting the Supervisor
-----------------------
On the server where you want to run the Xinference supervisor, run the following command:

.. code-block:: bash

  xinference-supervisor -H "${supervisor_host}"

Replace ${supervisor_host} with the actual host of your supervisor server.

Starting the Workers
--------------------

On each of the other servers where you want to run Xinference workers, run the following command:

.. code-block:: bash

  xinference-worker -e "http://${supervisor_host}:9997"

Once Xinference is running, an endpoint will be accessible for model management via CLI or Xinference client.


Using Xinference With Docker
=============================

To start Xinference in a Docker container, run the following command:

Run On Nvidia GPU Host
-----------------------

.. code-block:: bash

  docker run -p 9997:9997 --rm --gpus all apecloud/xinference:latest-amd64

Run On CPU Only Host
-----------------------

.. code-block:: bash

  docker run -p 9997:9997 --rm apecloud/xinference:latest-cpu


Using Xinference On Kubernetes
==============================

To use Xinference on Kubernetes, `KubeBlocks <https://kubeblocks.io/>`_ is required to help the installation.

The following steps assume Kubernetes is already installed.

1. Download cli tool kbcli for KubeBlocks, see `install kbcli <https://kubeblocks.io/docs/preview/user_docs/installation/install-with-kbcli/install-kbcli/>`_.

Make sure kbcli version is at least v0.7.1.

2. Install KubeBlocks using kbcli command, see `install KubeBlocks with kbcli <https://kubeblocks.io/docs/preview/user_docs/installation/install-with-kbcli/install-kubeblocks-with-kbcli/>`_.

3. Enable Xinference addon, run the following command:

.. code-block:: bash

  kbcli addon enable xinference

4. Use kbcli to start Xinference cluster, run the following command:

.. code-block:: bash

  kbcli cluster create xinference

If the Kubernetes node doesn't have GPU on it, run the command with extra flag:

.. code-block:: bash

  kbcli cluster create xinference --cpu-mode

Use -h to read the help documentation for more options:

.. code-block:: bash

  kbcli cluster create xinference -h