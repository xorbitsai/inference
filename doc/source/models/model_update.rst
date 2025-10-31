.. _model_update:

============
Model Update
============
.. versionadded:: v1.12.0

This section briefly introduces two common operations on the "Launch Model" page: updating model lists and adding models. They correspond to the "Type Selection + Update" and "Add Model" buttons at the top of the page, facilitating quick refresh of models of a certain type or adding new models locally.

Update Models (Launch Model Page)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Operation Location: "Type Selection" dropdown and "Update" button at the top right of the page.
- Usage:
1. Select a model type from the dropdown (such as llm, embedding, rerank, image, audio, video).
2. Click the "Update" button, the page will send an update request to the backend, then automatically jump to the corresponding Tab and refresh the model list of that type.

Add Model (Launch Model Page)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Operation Location: "Add Model" button at the top right of the page.
- Usage:
1. Click "Add Model" to open the add dialog.
2. Complete the model addition process in the dialog.
3. After successful addition, the page will jump to the corresponding type Tab to immediately view the latest model list.

Xinference Models Hub User Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overview
--------

Xinference Models Hub is a full-stack platform for managing and sharing models.
It provides a comprehensive solution for model registration, browsing, review workflows, and collaborative model management.

Quick Start
-----------

User Registration and Login
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Registration**

1. Open the website registration page
2. Fill in the necessary information and submit

**Login**

1. Open the website login page
2. After successful login, you will be redirected to the model list page

**Password Reset**

1. Click the "Forgot Password" link on the login page
2. Follow the instructions in the email to reset your password

**Logout**

1. Click the avatar in the top right corner of the page
2. Select "Logout" from the dropdown menu

Core Features
-------------

Browse Models
^^^^^^^^^^^^^

**Model List (Homepage)**

* **Function:** Browse available models, click any model to view details
* **Location:** "Models" menu in the website navigation bar

.. note::
   Some advanced models are only visible to authorized users.

**Model Details and Documentation**

* **Function:** View detailed information about models
* **Default Display:** "README" tab - view model description, usage instructions, and notes
* **Other Tabs:** Settings (authorized users), review status

User Center
^^^^^^^^^^^

* **Function:** View and manage personal information
* **Location:** Click the avatar in the top right corner, select "User Center"
* **Content:** Personal profile settings

Model Management (Authorized Users)
-----------------------------------

Model Registration
^^^^^^^^^^^^^^^^^^

* **Function:** Submit new models to the platform
* **Location:** Click the avatar in the top right corner, select "Model Registration"
* **Required Permissions:**

  * **Private Models:** Model registration permission
  * **Public Models:** Public model registration permission
  * **Enterprise Models:** Enterprise model registration permission

**Operation Process:**

1. Fill in basic model information
2. Fill in Readme (can be automatically obtained by clicking the Get Readme button)
3. Submit (to register public models, enable the Public Model parameter)

**Notes:**

  * Regular users can only register private models
  * Public model registration requires review, and can be used publicly after approval (no review needed if you have public model registration permission)
  * Enterprise model registration requires enabling the Public Model parameter first

My Models
^^^^^^^^^

* **Function:** View models associated with your account (models you registered)
* **Location:** Click the avatar in the top right corner, select "My Models"
* **Required Permissions:**

  * **Private Models:** Model registration permission
  * **Public Models:** Model registration permission
  * **Enterprise Models:** Model registration permission

Model Maintenance
^^^^^^^^^^^^^^^^^

* **Function:** Modify and manage existing models
* **Location:** Click the "Settings" icon on the model details page

* **Permission Requirements:**

  * **Private Models:** Model ownership or any public model management permission
  * **Advanced Models:** Advanced model update, delete, or expiration permission
  * **Public Models:** Public model update, delete, or expiration permission

**Notes:**

  * Updating JSON or modifying expiration attributes of public models will automatically create a PR to the xorbitsai/inference repository

Review Workflow
^^^^^^^^^^^^^^^

**For Model Submitters:**

1. Submit models for review
2. Check review status on the model details page
3. Make modifications based on reviewer feedback if needed

**For Reviewers:**

* **Required Permissions:** Model review list permission, model review permission

**Operation Process:**

1. Enter the review queue page
2. Evaluate model quality and compliance
3. Approve or reject and provide feedback