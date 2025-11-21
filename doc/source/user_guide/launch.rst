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

.. versionadded:: v1.14.0

Introduce a new environment variable:

.. code-block:: bash

    XINFERENCE_ENABLE_SINGLE_GPU_MULTI_REPLICA

Control whether to enable the single GPU multi-copy feature
Default value: 1

Key Features: Four new launch strategies
~~~~~~~~~~

Memory-Aware Strategy
^^^^^^^^

**Strategy Name:** `memory_aware` (default)

**Description:**
Advanced memory-aware allocation with intelligent GPU reuse.
Estimates model memory requirements using formula: `model_size(GB) × 1024 × 1.5`,
adjusts for quantization, and checks actual GPU availability before allocation.

Packing-First Strategy
^^^^^^^^

**Strategy Name:** `packing_first`

**Description:**
Prioritizes filling GPUs with the most available memory before moving to the next.
Optimizes for GPU utilization by consolidating instances.

Spread-First Strategy
^^^^^^^^

**Strategy Name:** `spread_first`

**Description:**
Distributes replicas across available GPUs using round-robin. Promotes load balancing and even resource utilization. Still sorts by available memory to prioritize better GPUs.

Quota-Aware Strategy
^^^^^^^^

**Strategy Name:** `quota_aware`

**Description:**
Restricts allocation to a specified GPU quota, then applies spread-first distribution within that quota. Ideal for multi-tenant resource isolation and cost allocation.

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
