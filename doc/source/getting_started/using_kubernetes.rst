.. _using_kubernetes:

########################
Xinference on Kubernetes
########################

************
Helm Support
************
Xinference provides a method for installation in a Kubernetes cluster via ``Helm`` .


Prerequisites
=============
* You have a fully functional Kubernetes cluster.
* Enable GPU support in Kubernetes, refer to `here <https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/>`_.
* ``Helm`` is correctly installed.


Steps
=====
#. Add xinference helm repo.

    .. code-block:: bash

      helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

#. Update xinference helm repo indexes and query versions.

    .. code-block:: bash

      helm repo update xinference
      helm search repo xinference/xinference --devel --versions

#. Install

    .. code-block:: bash

      helm install xinference xinference/xinference -n xinference --version <helm_charts_version>


Customized Installation
=======================
The installation method mentioned above sets up a Xinference cluster similar to a single-machine setup,
with only one worker and all startup parameters at their default values.
However, this is usually not the desired setup.

Below are some common custom installation configurations.

#. I need to download models from ``ModelScope``.

    .. code-block:: bash

      helm install xinference xinference/xinference -n xinference --version <helm_charts_version> --set config.model_src="modelscope"

#. I want to use cpu image of xinference (or use any other version of xinference images).

    .. code-block:: bash

      helm install xinference xinference/xinference -n xinference --version <helm_charts_version> --set config.xinference_image="<xinference_docker_image>"

#. I want to have 4 Xinference workers, with each worker managing 4 GPUs.

    .. code-block:: bash

      helm install xinference xinference/xinference -n xinference --version <helm_charts_version> --set config.worker_num=4 --set config.gpu_per_worker="4"

The above installation method is based on Helm ``--set`` option.
For more complex custom installations, such as multiple workers with shared storage,
it is highly recommended to use your own ``values.yaml`` file with Helm ``-f`` option for installation.

The default ``values.yaml`` file is located `here <https://github.com/xorbitsai/xinference-helm-charts/blob/main/charts/xinference/values.yaml>`_.
Some examples can be found `here <https://github.com/xorbitsai/xinference-helm-charts/tree/main/examples>`_.


******************
KubeBlocks Support
******************
You can also install Xinference in Kubernetes using the third-party ``KubeBlocks``.
This method is not maintained by Xinference and does not guarantee timely updates or availability.
Please refer to the documentation at `here <https://kubeblocks.io/docs/preview/user_docs/kubeblocks-for-xinference/manage-xinference>`_.
