.. _launch:

============================
Model Launching Instructions
============================

This document aims to provide a functional overview of model launching.

Replica
=======

Replicas specify the number of model instances to load. For example, if you have two GPUs and each can host one replica of the model,
you can set the replica count to 2. This way, two identical instances of the model will be distributed across the two GPUs.
Xinference automatically load-balances requests to ensure even distribution across multiple GPUs.
Meanwhile, users see it as a single model, which greatly improves overall resource utilization.

Traditional Multi-Instance Deployment：

When you have multiple GPU cards, each capable of hosting one model instance, you can set the number of instances equal to the number of GPUs. For example:

- 2 GPUs, 2 instances: Each GPU runs one model instance
- 4 GPUs, 4 instances: Each GPU runs one model instance

.. versionadded:: v1.11.1

Introduce a new environment variable:

.. code-block:: bash

    XINFERENCE_ENABLE_SINGLE_GPU_MULTI_REPLICA

Control whether to enable the single GPU multi-copy feature
Default value: 1

New Feature: Smart Replica Deployment

1. Single GPU Multi-Replica

New Support: Run multiple model replicas even with just one GPU.

- Scenario: You have 1 GPU with sufficient VRAM
- Configuration: Replica Count = 3, GPU Count = 1
- Result: 3 model instances running on the same GPU, sharing GPU resources

2. Hybrid GPU Allocation

Smart Allocation: Number of replicas may differ from GPU count; system intelligently distributes

- Scenario: You have 2 GPUs and need 3 replicas
- Configuration: Replicas=3, GPUs=2
- Result: GPU0 runs 2 instances, GPU1 runs 1 instance

Set Environment Variables
=========================

.. versionadded:: v1.8.1

Sometimes, we want to specify environment variables for a particular model at runtime.
Since v1.8.1, Xinference provides the capability to configure these individually without needing to set them before starting Xinference.

For Web UI.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/launch_env.png" style="background-color: transparent", width="95%">

When using the command line, use ``--env`` to specify an environment variable.

Example usage:

.. code-block:: bash

  xinference launch xxx --env A 0 --env B 1

Take vLLM as an example: it has versions V1 and V0, and by default, it automatically determines which version to use.
If you want to force the use of V0 by setting ``VLLM_USE_V1=0`` when launching a model, you can specify this during model launching.

Configuring Model Virtual Environment
=====================================

.. versionadded:: v1.8.1

For this part, please refer to :ref:`toggling virtual environments and customizing dependencies <model_launching_virtualenv>`.
