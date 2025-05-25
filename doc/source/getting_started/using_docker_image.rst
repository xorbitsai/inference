.. _using_docker_image:

=======================
Xinference Docker Image
=======================

Xinference provides official images for use on Dockerhub.


Prerequisites
=============
* The image can only run in an environment with GPUs and CUDA installed, because Xinference in the image relies on Nvidia GPUs for acceleration.
* CUDA must be successfully installed on the host machine. This can be determined by whether you can successfully execute the ``nvidia-smi`` command.
* The CUDA version in the docker image is ``12.4``, and the CUDA version on the host machine should be ``12.4`` or above, and the NVIDIA driver version should be ``550`` or above.
* Ensure `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ installed.


Docker Image
============
The official image of Xinference is available on DockerHub in the repository ``xprobe/xinference``.
Available tags include:

* ``nightly-main``: This image is built daily from the `GitHub main branch <https://github.com/xorbitsai/inference>`_ and generally does not guarantee stability.
* ``v<release version>``: This image is built each time a Xinference release version is published, and it is typically more stable.
* ``latest``: This image is built with the latest Xinference release version.
* For CPU version, add ``-cpu`` suffix, e.g. ``nightly-main-cpu``.


Dockerfile for custom build
===========================
If you need to build the Xinference image according to your own requirements, the source code for the Dockerfile is located at `xinference/deploy/docker/Dockerfile <https://github.com/xorbitsai/inference/tree/main/xinference/deploy/docker/Dockerfile>`_ for reference.
Please make sure to be in the top-level directory of Xinference when using this Dockerfile. For example:

.. code-block:: bash

   git clone https://github.com/xorbitsai/inference.git
   cd inference
   docker build --progress=plain -t test -f xinference/deploy/docker/Dockerfile .


Image usage
===========
You can start Xinference in the container like this, simultaneously mapping port 9997 in the container to port 9998 on the host, enabling debug logging, and downloading models from modelscope.

.. code-block:: bash

   docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 --gpus all xprobe/xinference:v<your_version> xinference-local -H 0.0.0.0 --log-level debug


.. warning::
    * The option ``--gpus`` is essential and cannot be omitted, because as mentioned earlier, the image requires the host machine to have a GPU. Otherwise, errors will occur.
    * The ``-H 0.0.0.0`` parameter after the ``xinference-local`` command cannot be omitted. Otherwise, the host machine may not be able to access the port inside the container.
    * You can add multiple ``-e`` options to introduce multiple environment variables.


Certainly, if you prefer, you can also manually enter the docker container and start Xinference in any desired way.

.. note::

   For multiple GPUs, make sure to set the shared memory size, for example: `docker run --shm-size=128g ...`


Mount your volume for loading and saving models
===============================================
The image does not contain any model files by default, and it downloads the models into the container.
Typically, you would need to mount a directory on the host machine to the docker container, so that Xinference can download the models onto it, allowing for reuse.
In this case, you need to specify a volume when running the Docker image and configure environment variables for Xinference:

.. code-block:: bash

   docker run -v </on/your/host>:</on/the/container> -e XINFERENCE_HOME=</on/the/container> -p 9998:9997 --gpus all xprobe/xinference:v<your_version> xinference-local -H 0.0.0.0


The principle behind the above command is to mount the specified directory from the host machine into the container, and then set the ``XINFERENCE_HOME`` environment variable to point to that directory inside the container.
This way, all downloaded model files will be stored in the directory you specified on the host machine.
You don't have to worry about losing them when the Docker container stops, and the next time you run it, you can directly use the existing models without the need for repetitive downloads.

If you downloaded the model using the default path on the host machine, and since the xinference cache directory
stores the model using symbolic links, you need to mount the directory where the original file is located into the container as well.
For example, if you are using HuggingFace and Modelscope as model hub, you would need to mount the corresponding
directories into the container. Generally, the cache directories for HuggingFace and Modelscope are located
at <home_path>/.cache/huggingface and <home_path>/.cache/modelscope. The command would be like:

.. code-block:: bash

   docker run \
     -v </your/home/path>/.xinference:/root/.xinference \
     -v </your/home/path>/.cache/huggingface:/root/.cache/huggingface \
     -v </your/home/path>/.cache/modelscope:/root/.cache/modelscope \
     -p 9997:9997 \
     --gpus all \
     xprobe/xinference:v<your_version> \
     xinference-local -H 0.0.0.0


