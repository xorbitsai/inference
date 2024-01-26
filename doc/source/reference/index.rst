.. _reference_index:

=============
API Reference
=============


Client
~~~~~~~
.. autosummary::
   :toctree: generated/

   xinference.client.Client

   xinference.client.Client.describe_model
   xinference.client.Client.get_model
   xinference.client.Client.get_model_registration
   xinference.client.Client.launch_model
   xinference.client.Client.list_model_registrations
   xinference.client.Client.list_models
   xinference.client.Client.register_model
   xinference.client.Client.terminate_model
   xinference.client.Client.unregister_model


Model Handles
~~~~~~~~~~~~~


ChatglmCppChatModelHandle
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.ChatglmCppChatModelHandle

   xinference.client.handlers.ChatglmCppChatModelHandle.chat


ChatModelHandle
^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.ChatModelHandle

   xinference.client.handlers.ChatModelHandle.chat
   xinference.client.handlers.ChatModelHandle.generate


EmbeddingModelHandle
^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.EmbeddingModelHandle

   xinference.client.handlers.EmbeddingModelHandle.create_embedding


GenerateModelHandle
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.GenerateModelHandle

   xinference.client.handlers.GenerateModelHandle.generate


ImageModelHandle
^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.ImageModelHandle

   xinference.client.handlers.ImageModelHandle.text_to_image


AudioModelHandle
^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.AudioModelHandle

   xinference.client.handlers.AudioModelHandle.transcriptions
   xinference.client.handlers.AudioModelHandle.translations
