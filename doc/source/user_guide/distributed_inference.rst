.. _distributed_inference:

#####################
Distributed Inference
#####################
Some language models including **DeepSeek V3**, **DeepSeek R1**, etc are too large to fit into GPus
on a single machine, Xinference supported running these models across multiple machines.

.. note::
    This feature is added in v1.3.0.

*****************
Supported Engines
*****************
Now, Xinference supported below engines to run models across workers.

* :ref:`SGLang <sglang_backend>` (supported in v1.3.0)
* :ref:`vLLM <vllm_backend>` (supported in v1.4.1)

Upcoming supports. Below engine will support distributed inference soon:

* :ref:`MLX <mlx_backend>`

*****
Usage
*****
First you need at least 2 workers to support distributed inference.
Refer to :ref:`running Xinference in cluster <distributed_getting_started>`
to create a Xinference cluster including supervisor and workers.

Then if are using web UI, choose expected machines for ``worker count`` in the optional configurations,
if you are using command line, add ``--n-worker <machine number>`` when launching a model.
The model will be launched across multiple workers accordingly.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/distributed_inference.png" style="background-color: transparent", width="77%">

``GPU count`` on web UI, or ``--n-gpu`` for command line now mean GPUs count per worker if you are using distributed inference.
