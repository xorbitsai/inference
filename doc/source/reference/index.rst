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
   xinference.client.Client.get_launch_model_progress
   xinference.client.Client.cancel_launch_model
   xinference.client.Client.get_instance_info
   xinference.client.Client.launch_model
   xinference.client.Client.list_model_registrations
   xinference.client.Client.list_models
   xinference.client.Client.list_cached_models
   xinference.client.Client.list_deletable_models
   xinference.client.Client.confirm_and_remove_model
   xinference.client.Client.query_engine_by_model_name
   xinference.client.Client.register_model
   xinference.client.Client.terminate_model
   xinference.client.Client.abort_request
   xinference.client.Client.vllm_models
   xinference.client.Client.login
   xinference.client.Client.get_workers_info
   xinference.client.Client.get_supervisor_info
   xinference.client.Client.get_progress
   xinference.client.Client.abort_cluster
   xinference.client.Client.unregister_model


Model Handles
~~~~~~~~~~~~~


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


RerankModelHandle
^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.restful.restful_client.RESTfulRerankModelHandle

   xinference.client.restful.restful_client.RESTfulRerankModelHandle.rerank


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
   xinference.client.handlers.AudioModelHandle.speech


FlexibleModelHandle
^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.restful.restful_client.RESTfulFlexibleModelHandle

   xinference.client.restful.restful_client.RESTfulFlexibleModelHandle.infer


VideoModelHandle
^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree: generated/

   xinference.client.handlers.VideoModelHandle

   xinference.client.handlers.VideoModelHandle.text_to_video
