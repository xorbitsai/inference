.. _distributed_inference:

#####################
Distributed Inference
#####################
Some language models including **DeepSeek V3**, **DeepSeek R1**, etc are too large to fit into GPus
on a single machine, Xinference supported running these models across multiple machines.

.. versionadded:: v1.3.0

*****************
Supported Engines
*****************
Now, Xinference supported below engines to run models across workers.

* :ref:`SGLang <sglang_backend>` (supported in v1.3.0)
* :ref:`vLLM <vllm_backend>` (supported in v1.4.1)
* :ref:`MLX <mlx_backend>` (supported in v1.7.1), MLX distributed currently does not support all models.
  The following model types are supported at this time. If you have additional requirements,
  feel free to submit a GitHub issue at `https://github.com/xorbitsai/inference/issues <https://github.com/xorbitsai/inference/issues>`_ to request support.

  - DeepSeek v3 and R1
  - Qwen2.5-instruct and the models have the same model architectures.
  - Qwen3 and the models have the same model architectures.
  - Qwen3-moe and the models have the same model architectures.


*****
Usage
*****
First you need at least 2 workers to support distributed inference.
Refer to :ref:`running Xinference in cluster <distributed_getting_started>`
to create a Xinference cluster including supervisor and workers.

vLLM (v0.11.0+) note:
Starting from vLLM v0.11.0, distributed deployment with vLLM requires Xinference >= v1.17.1.
In addition to setting ``--n-worker`` as before, you must also set
``tensor_parallel_size=2`` and ``pipeline_parallel_size=1`` when launching the model.

Then if are using web UI, choose expected machines for ``worker count`` in the optional configurations,
if you are using command line, add ``--n-worker <machine number>`` when launching a model.
The model will be launched across multiple workers accordingly.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/distributed_inference.png" style="background-color: transparent", width="77%">

``GPU count`` on web UI, or ``--n-gpu`` for command line now mean GPUs count per worker if you are using distributed inference.
