.. _model_virtual_env:

==========================
Model Virtual Environments
==========================

.. note::

  New added in v1.5.0.

Background
##########

Some models are no longer maintained after their release, and the versions of the libraries they depend on remain outdated.
For example, the ``GOT-OCR2`` model still relies on ``transformers`` version 4.37.2. If this library is updated to a newer version,
the model can no longer function properly. On the other hand, many newer models require the latest version of ``transformers``.
This version mismatch leads to dependency conflicts.

Solution
########

To address this issue, we have introduced the **Model Virtual Environment** feature.

Install requirements for this functionality via

.. code-block:: bash

    # all
    pip install 'xinference[all]'
    # or virtualenv
    pip install 'xinference[virtualenv]'

Enable by setting environment variable ``XINFERENCE_ENABLE_VIRTUAL_ENV=1``.

Example usage:

.. code-block:: bash

  # For command line
  XINFERENCE_ENABLE_VIRTUAL_ENV=1 xinference-local ...

  # For Docker
  docker run -e XINFERENCE_ENABLE_VIRTUAL_ENV=1 ...

.. warning::

  This feature requires internet access or a self-hosted PyPI mirror.

  Xinference will by default inherit the config for current pip.

.. note::

  The model virtual environment feature is disabled by default (i.e., XINFERENCE_ENABLE_VIRTUAL_ENV is set to 0).

  It will be enabled by default starting from Xinference v2.0.0.

When enabled, Xinference will automatically create a dedicated virtual environment for each model when it is loaded,
and install its specific dependencies there. This prevents dependency conflicts between models,
allowing them to run in isolation without affecting one another.

Supported Models
################

Currently, this feature supports the following models:

* :ref:`GOT-OCR2 <models_builtin_got-ocr2_0>`
* :ref:`Qwen2.5-omni <models_llm_qwen2.5-omni>`
* ... (New models since v1.5.0 will all consider to add support)

Storage Location
################

By default, the model’s virtual environment is stored under path:

* Before v1.6.0: :ref:`XINFERENCE_HOME <environments_xinference_home>` / virtualenv / {model_name}
* Since v1.6.0: :ref:`XINFERENCE_HOME <environments_xinference_home>` / virtualenv / v2 / {model_name}
