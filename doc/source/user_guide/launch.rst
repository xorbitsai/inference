.. _launch:

==========================
Model Loading Instructions
==========================

This document aims to provide a functional overview of model loading.

Replica
=======

Replicas specify the number of model instances to load. For example, if you have two GPUs and each can host one replica of the model,
you can set the replica count to 2. This way, two identical instances of the model will be distributed across the two GPUs.
Xinference automatically load-balances requests to ensure even distribution across multiple GPUs.
Meanwhile, users see it as a single model, which greatly improves overall resource utilization.

Set Environment Variables
=========================

Sometimes, we want to specify environment variables for a particular model at runtime.
Xinference provides the capability to configure these individually without needing to set them before starting Xinference.

For Web UI.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/launch_env.png" style="background-color: transparent", width="95%">

When using the command line, use ``--env`` to specify an environment variable.

Example usage:

.. code-block:: bash

  xinference launch xxx --env A 0 --env B 1

Take vLLM as an example: it has versions V1 and V0, and by default, it automatically determines which version to use.
If you want to force the use of V0 by setting ``VLLM_USE_V1=0`` when loading a model, you can specify this during model loading.
