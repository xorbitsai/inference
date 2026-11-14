.. _model_update:

============
Model Update
============
.. versionadded:: v1.13.0

This section briefly introduces a common operation on the "Launch Model" page: updating the model list. It corresponds to the "Type Selection + Update" button at the top of the page, which is used to quickly refresh models of a specific type.

Xinference reads model definitions directly from the `public GitHub repository <https://github.com/xorbitsai/inference/tree/main/xinference/model>`_. Use **Update** to refresh a category or **Update one model** to update one model. These actions update model definitions. Model weights are fetched from the source specified by each definition when needed.

.. raw:: html

    <img class="align-center" alt="model update interface" src="../_static/model_update.png" style="background-color: transparent", width="95%">

Update Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Operation Location: "Type Selection" dropdown and "Update" button at the top right of the page.
- Usage:
1. Select a model type from the dropdown (such as llm, embedding, rerank, image, audio, video).
2. Click the "Update" button, the page will send an update request to the backend, then automatically jump to the corresponding Tab and refresh the model list of that type.
