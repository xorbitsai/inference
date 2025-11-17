.. _xinference_models_hub:

=====================
Xinference Models Hub
=====================
.. versionadded:: v1.13.0

Overview
########

The `Xinference Models Hub <https://model.xinference.io>`_ is Xinference’s unified platform for model management and collaboration. It provides end-to-end support for model browsing, registration, review, updates, and collaborative maintenance, serving both regular users and model maintainers.

Before the introduction of the Models Hub, model JSON files were manually submitted and modified directly through PRs in the open-source repository (`xorbitsai/inference`). This approach resulted in uncontrolled versioning, long iteration cycles, and delays in delivering updated models—since model updates were tied to product release cycles, users could not obtain new models promptly.

With centralized online model management, the Xinference Models Hub requires that **all model information—including metadata, parameters, and the README—be edited within the platform**. Based on modifications made by :ref:`model maintainers <model_maintainer_guide>`, the system **automatically generates and submits PRs** to the `xorbitsai/inference` repository, ensuring a standardized, automated, and traceable workflow that eliminates inconsistencies caused by manual edits.

Users can obtain the latest model list at any time through the :ref:`model_update` feature, significantly improving model delivery efficiency and overall experience.

User Guide for Regular Users
############################

This section introduces the basic features available to regular registered users.

**Audience:** Regular users without model registration or maintenance permissions.

Core Features
^^^^^^^^^^^^^

Regular users can browse public models without logging in:

* **Access:** `Xinference Models Hub <https://model.xinference.io>`_

Browse Models
~~~~~~~~~~~~~

* **Function:** View all publicly available models  
* **Location:** Navigation bar → “Models”  
* **Model Details:** The default tab is “README,” which includes model description, usage guide, and important notes  

.. note::
   Certain advanced or enterprise-level models are visible only to authorized users.

.. _model_maintainer_guide:

Guide for Model Maintainers
############################

This section describes the features available to users with model registration or maintenance permissions, including model registration, updates, and review workflows.

**Audience:** Users with model registration or maintenance permissions.  
To become a model maintainer, you may `contact us <https://xorbits.cn/assets/images/wechat_work_qr.png>`_.

Core Features (Login Required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model maintainers have access to the following advanced features in addition to the capabilities available to regular users.

User Center
~~~~~~~~~~~

* **Function:** View and manage personal information  
* **Location:** Top-right avatar → “User Center”  
* **Account Management:** Update profile, email, and other information  
* **Token Management:** Configure a personal GitHub Token for model submissions or updates  

.. note::
   If no GitHub Token is configured, the system will use a default Token when generating PRs.

Model Registration
~~~~~~~~~~~~~~~~~~

* **Function:** Register new models and submit them for review  
* **Location:** After logging in → Top-right avatar → “Model Registration”

**Submission Steps:**

1. Fill in basic model information (name, engine, format, etc.)  
2. Edit the README (click “Get README” to auto-generate a template)  
3. Submit the model (enable the “Public Model” parameter if registering a public model)

.. note::
   When registering a public model, the system automatically creates a PR in the `xorbitsai/inference` repository.

My Models
~~~~~~~~~

* **Function:** View all models associated with the current account  
* **Location:** After logging in → Top-right avatar → “My Models”

Model Maintenance
~~~~~~~~~~~~~~~~~

* **Function:** Modify the model’s JSON or README  
* **Location:** Model details page → “Settings” icon  

.. note::
   When updating a public model, the system automatically creates a PR in the `xorbitsai/inference` repository.

Review Workflow
~~~~~~~~~~~~~~~

**Submitter Workflow:**

1. Submit a model  
2. Wait for review  
3. Revise based on reviewer feedback  
4. Updated PRs are generated automatically (for public models)

**Reviewer Permissions:** Access to the model review list and model review privileges.

**Reviewer Workflow:**

1. Enter the “Review List”  
2. Evaluate the model’s quality, completeness, and compliance  
3. Approve or reject, providing feedback as necessary
