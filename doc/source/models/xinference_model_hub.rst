.. _xinference_model_hub:

================================
Xinference Models Hub User Guide
================================
.. versionadded:: v1.13.0

Overview
########

`Xinference Models Hub <https://model.xinference.io>`_ is a full-stack platform for managing and sharing models, providing a complete solution for model registration, browsing, review workflows, and collaborative model management.

User Guide for Regular Users
############################

This section introduces the main features available to regular registered users, including model browsing and personal center management.

**Audience:** Regular registered users without model maintenance or public model registration permissions.

Core Features
^^^^^^^^^^^^^

Browse Models
~~~~~~~~~~~~~

* **Function:** View available models and click to see details  
* **Location:** Navigation bar → “Models”  
* **Model Details Page:** The default tab is “README,” where you can view model descriptions, usage instructions, and important notes  

.. note::
   Some advanced models are only visible to authorized users.

User Center
~~~~~~~~~~~

* **Function:** View and manage personal information  
* **Location:** Click the avatar in the top-right corner → “User Center”  
* **Account Management:** Update personal profile, email, and other details  
* **Token Management:** Configure a GitHub Token used for model submissions or updates  

.. note::
   If no token is configured, the system will use a default token to create pull requests (PRs).

Workflow
~~~~~~~~

1. **Register an account** → Log in → Browse models  
2. **Reset password:** Click “Forgot Password” on the login page and follow the email instructions  
3. **Logout:** Click the avatar in the top-right corner → “Logout”

Guide for Model Maintainers
###########################

This section is for users with model registration or maintenance permissions. It introduces model registration, maintenance, and the review workflow.

**Audience:** Users with model registration or maintenance permissions.  
If you wish to become a model maintainer, you can `contact us <https://xorbits.cn/assets/images/wechat_work_qr.png>`_.

Core Features
^^^^^^^^^^^^^

Includes all features available to regular users, plus the following advanced functions.

Model Registration
~~~~~~~~~~~~~~~~~~

* **Function:** Submit new models
* **Location:** Click the avatar in the top-right corner → “Model Registration”

**Operation Steps:**

1. Fill in basic model information
2. Complete the README (click “Get README” to auto-generate)
3. Submit (for public models, enable the “Public Model” parameter)

.. note::
    When registering a public model, the system will automatically create a PR in the `xorbitsai/inference` repository.
    If the user has configured a GitHub Token in their personal settings, the system will use that Token to submit the PR; otherwise, the default Token will be used.

**Notes:**

* Enterprise model registration requires enabling the “Public Model” parameter first.

My Models
~~~~~~~~~~

* **Function:** View models associated with your account  
* **Location:** Click the avatar in the top-right corner → “My Models”

Model Maintenance
~~~~~~~~~~~~~~~~~

* **Function:** Modify and manage existing models  
* **Location:** Model Details → “Settings” icon  

.. note::
   When updating the JSON of a public model or modifying expiration attributes, the system automatically creates a PR in the `xorbitsai/inference` repository.  

Review Workflow
~~~~~~~~~~~~~~~

**For Submitters:**

1. Submit a model  
2. Check the review status  
3. Modify based on reviewer feedback

**For Reviewers:**

* **Required Permissions:** Model review list access and model review permissions  

**Operation Steps:**

1. Enter the review list  
2. Evaluate model quality and compliance  
3. Approve or reject, providing feedback as needed
