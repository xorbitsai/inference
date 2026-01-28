.. _model_virtual_env:
.. _virtualenv:

==========================
Model Virtual Environments
==========================

.. versionadded:: v1.5.0

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

  Starting from **Xinference v2.0**, the model virtual environment feature is
  enabled by default (i.e., ``XINFERENCE_ENABLE_VIRTUAL_ENV`` defaults to ``1``).

  To disable it globally, set ``XINFERENCE_ENABLE_VIRTUAL_ENV=0`` when starting Xinference.

When enabled, Xinference will automatically create a dedicated virtual environment for each model when it is loaded,
and install its specific dependencies there. This prevents dependency conflicts between models,
allowing them to run in isolation without affecting one another.

Using Virtual Environments (v2.0)
#################################

Global toggle
~~~~~~~~~~~~~

Virtual environments are enabled by default starting from v2.0. You can still override this globally:

.. code-block:: bash

  # Enable globally (default)
  XINFERENCE_ENABLE_VIRTUAL_ENV=1 xinference-local -H 0.0.0.0 -p 9997

  # Disable globally
  XINFERENCE_ENABLE_VIRTUAL_ENV=0 xinference-local -H 0.0.0.0 -p 9997

Per-model override at launch time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can override the global setting when launching a model:

.. code-block:: bash

  # Force enable for this model
  xinference launch -n qwen2.5-instruct --model-engine transformers --enable-virtual-env

  # Force disable for this model
  xinference launch -n qwen2.5-instruct --model-engine transformers --disable-virtual-env

Add or override packages at launch time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--virtual-env-package`` (or ``-vp``) multiple times:

.. code-block:: bash

  xinference launch -n qwen2.5-instruct --model-engine transformers \
    --virtual-env-package transformers==4.46.3 \
    --virtual-env-package accelerate==0.33.0

If you specify a package that already exists in the model's default virtualenv package list,
your version replaces the default instead of being appended.


Storage Location
################

By default, the model’s virtual environment is stored under path:

* Before v1.6.0: :ref:`XINFERENCE_HOME <environments_xinference_home>` / virtualenv / {model_name}
* From v1.6.0 to v1.13.0: :ref:`XINFERENCE_HOME <environments_xinference_home>` / virtualenv / v2 / {model_name}
* Since v1.14.0: :ref:`XINFERENCE_HOME <environments_xinference_home>` / virtualenv / v3 / {model_name} / {python_version}
* Since v2.0: :ref:`XINFERENCE_HOME <environments_xinference_home>` / virtualenv / v4 / {model_name} / {model_engine} / {python_version}

Experimental Feature
####################

.. _skip_installed_libraries:

Skip Installed Libraries
------------------------

.. versionadded:: v1.8.1

   This feature requires ``xoscar >= 0.7.12``, which is the minimum Xoscar version required for Xinference v1.8.1.

``xinference`` uses the ``uv`` tool to create virtual environments, with the current Python **system site-packages** set as the base environment.
By default, ``uv`` **does not check for existing packages in the system environment** and reinstalls all dependencies in the virtual environment.
This ensures better isolation from system packages but can result in redundant installations, longer setup times, and increased disk usage.

Starting from ``v1.8.1``, an **experimental feature** is available:
by setting the environment variable ``XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED=1``, ``uv`` will **skip packages already available in system site-packages**.

.. note::

    The feature is currently disabled but will be enabled by default in ``v2.0.0``.

Advantages
~~~~~~~~~~

- Avoid redundant installations of large dependencies (e.g., ``torch`` + ``CUDA``).
- Speed up virtual environment creation.
- Reduce disk usage.

Usage
~~~~~

.. code-block:: bash

   # Enable experimental feature

   # For command line
   XINFERENCE_ENABLE_VIRTUAL_ENV=1 XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED=1 xinference-local ...
   # For docker
   docker run -e XINFERENCE_ENABLE_VIRTUAL_ENV=1 -e XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED=1 ...

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

Using the ``CosyVoice 0.5B`` model as an example:

**Without this feature enabled**::

    Installed 98 packages in 187ms
     + aiohappyeyeballs==2.6.1
     + aiohttp==3.12.13
     ...
     + torch==2.7.1
     ...
     + yarl==1.20.1
     + zipp==3.23.0

**With this feature enabled**::

    Installed 7 packages in 12ms
     + diffusers==0.29.0
     + hf-xet==1.1.5
     + huggingface-hub==0.33.2
     + importlib-metadata==8.7.0
     + pillow==11.3.0
     + typing-extensions==4.14.0
     + urllib3==2.5.0


.. _model_launching_virtualenv:

Model Launching: Toggle Virtual Environments and Customize Dependencies
-----------------------------------------------------------------------

.. versionadded:: v1.8.1

Starting from v1.8.1, we support toggling the virtual environment for individual model launching,
as well as overriding the model's default settings with custom package dependencies.

Toggle Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

When loading a model, you can specify whether to enable the model's virtual environment.
If not specified, the setting will follow the environment variable configuration.

For the Web UI, this can be toggled on or off through the optional settings switch.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/model_virtual_env.png" style="background-color: transparent", width="95%">

For command-line loading, use the ``--enable-virtual-env`` option to enable the virtual environment, or ``--disable-virtual-env`` to disable it.

Example usage:

.. code-block:: bash

  xinference launch xxx --enable-virtual-env

Set Virtual Environment Package Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For supported models, Xinference has already defined the package dependencies and version requirements within the virtual environment.
However, if you need to specify particular versions or install additional dependencies, you can manually provide them during model loading.

In the Web UI, you can add custom dependencies by clicking the plus icon in the same location as the virtual environment toggle.

For the command line, use ``--virtual-env-package`` or ``-vp`` to specify a single package version.

Example usage:

.. code-block:: bash

  xinference launch xxx --virtual-env-package transformers==4.54.0

In addition to the standard way of specifying package dependencies, such as ``transformers==xxx``, Xinference also supports some extended syntax.

* ``#system_xxx#``: Using the same version as the system site packages, such as ``#system_numpy#``,
  ensures that the installed package matches the system site package version of numpy. This helps prevent dependency conflicts.

Authoring Custom Models (JSON)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When registering a custom model, you can define a ``virtualenv`` block in the model JSON.
Starting from v2.0 (v4 flow), **engine-aware markers are recommended** so one JSON can cover
multiple engines.

Important rule:
If a new model supports a specific engine, you **must** include at least one package
entry for that engine in ``virtualenv.packages`` and attach a marker, for example
``#engine# == "vllm"``. Engine availability checks rely on these markers when
virtual environments are enabled.

Minimal virtualenv block (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

  {
    "virtualenv": {
      "packages": [
        "#transformers_dependencies# ; #engine# == \"transformers\"",
        "#vllm_dependencies# ; #engine# == \"vllm\"",
        "#sglang_dependencies# ; #engine# == \"sglang\"",
        "#llama_cpp_dependencies# ; #engine# == \"llama.cpp\"",
        "#mlx_dependencies# ; #engine# == \"mlx\"",
        "#system_numpy# ; #engine# == \"vllm\""
      ]
    }
  }

Field reference
^^^^^^^^^^^^^^^

- ``packages`` (required): list of pip requirement strings or markers.
- ``inherit_pip_config`` (default ``true``): inherit system pip configuration if present.
- ``index_url`` / ``extra_index_url`` / ``find_links`` / ``trusted_host``:
  pip index and mirror controls.
- ``index_strategy``: passed through to the virtualenv installer (used by some engines).
- ``no_build_isolation``: pip build isolation switch for tricky builds.

Engine placeholders
^^^^^^^^^^^^^^^^^^

Use wrapped placeholders to inject engine defaults:

- ``#vllm_dependencies#``
- ``#sglang_dependencies#``
- ``#mlx_dependencies#``
- ``#transformers_dependencies#``
- ``#llama_cpp_dependencies#``
- ``#diffusers_dependencies#``
- ``#sentence_transformers_dependencies#``

Markers and case
^^^^^^^^^^^^^^^^

Markers use ``#engine#`` or ``#model_engine#`` comparisons (case-sensitive).
Engine values are passed in lowercase internally, so prefer lowercase values,
for example ``#engine# == "vllm"`` or ``#engine# == "transformers"``.


.. _manage_virtual_enviroments:

Manage Virtual Enviroments
------------------------

.. versionadded:: v1.14.0

Xinference provides comprehensive virtual environment management for model dependencies,
allowing you to create isolated Python environments for each model with specific package requirements.

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/manage_virtual_envs1.png" style="background-color: transparent", width="95%">

.. raw:: html

    <img class="align-center" alt="actor" src="../_static/manage_virtual_envs2.png" style="background-color: transparent", width="95%">

Key Features
~~~~~~~~~~

**Multiple Python Version Support**:
Each model can have virtual environments
with different Python versions (e.g., Python 3.10.18, 3.11.5),
enabling compatibility with various model requirements.

**Isolated Dependencies**:
Each virtual environment contains its own set of packages,
preventing conflicts between different models' requirements.

Management Operations
~~~~~

**Listing Virtual Environments**:
View all virtual environments across your cluster,
filtered by model name or worker IP address.

**Creating Environments**:
Automatically created when launching models with enable_virtual_env=true.
The system detects your current Python version and creates an isolated
environment with the required packages.

**Removing Environments**:
Delete specific virtual environments by model name and optionally
Python version, or remove all environments for a model.
