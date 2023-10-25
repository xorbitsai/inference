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

  xinference --host 0.0.0.0 --port 9997


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
