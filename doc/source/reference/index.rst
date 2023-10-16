.. _reference_index:

=============
API Reference
=============
.. currentmodule:: xinference.client


Client
~~~~~~~
.. autosummary::
   :toctree: generated/

   Client

   Client.describe_model
   Client.get_model
   Client.get_model_registration
   Client.launch_model
   Client.list_model_registrations
   Client.list_models
   Client.register_model
   Client.terminate_model
   Client.unregister_model


Model Handles
~~~~~~~~~~~~~


ChatglmCppChatModelHandle
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.ChatglmCppChatModelHandle

   xinference.client.handlers.ChatglmCppChatModelHandle.chat
   xinference.client.handlers.ChatglmCppChatModelHandle.create_embedding


ChatModelHandle
^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.ChatModelHandle

   xinference.client.handlers.ChatModelHandle.chat
   xinference.client.handlers.ChatModelHandle.create_embedding
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

   xinference.client.handlers.GenerateModelHandle.create_embedding
   xinference.client.handlers.GenerateModelHandle.generate


ImageModelHandle
^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.ImageModelHandle

   xinference.client.handlers.ImageModelHandle.text_to_image
