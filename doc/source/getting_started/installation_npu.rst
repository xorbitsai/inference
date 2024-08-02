.. _installation_npu:


=================================
Installation Guide for Ascend NPU
=================================
Xinference can run on Ascend NPU, follow below instructions to install.


Installing PyTorch and Ascend extension for PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install PyTorch CPU version and corresponding Ascend extension.

Take PyTorch v2.1.0 as example.

  .. code-block:: bash

    pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

Then install `Ascend extension for PyTorch <https://github.com/Ascend/pytorch>`_.

  .. code-block:: bash

    pip3 install 'numpy<2.0'
    pip3 install decorator
    pip3 install torch-npu==2.1.0.post3

Running below command to see if it correctly prints the Ascend NPU count.

.. code-block:: bash

    python -c "import torch; import torch_npu; print(torch.npu.device_count())"

Installing Xinference
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip3 install xinference

Now you can use xinference according to :ref:`doc <using_xinference>`.
``Transformers`` backend is the only available engine supported for Ascend NPU for open source version.

Enterprise Support
~~~~~~~~~~~~~~~~~~
If you encounter any performance or other issues for Ascend NPU, please reach out to us
via `link <https://xorbits.io/community>`_.
