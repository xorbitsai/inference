.. _release_ntoes:

Release Notes
=============

.. _release_4_0_overview:

Xinference 4.0 overview
-----------------------

The v4.0.0 community release expands cache reuse, heterogeneous
deployment, and runtime model management. The highlights below summarize the
major changes; detailed notes for individual versions follow this overview.

* **Multi-engine Xavier and P/D.** SGLang and MLX replicas gain prefix KV
  cache sharing. P/D deployments can combine vLLM, SGLang, and MLX, including
  NVIDIA and Apple silicon workers. Native NIXL backends are available for
  vLLM and SGLang; vLLM Xavier adds direct GPU handoff and tiered history.
  See :ref:`user_guide_vllm_enhancement` and :ref:`user_guide_pd_separation`.
* **Reconfigure without reloading weights.** Opt-in GPU weight caching lets
  vLLM and SGLang rebuild the engine with updated supported parameters while
  retaining the weights. Current requests drain before the change, and the
  model UID stays the same. See :ref:`launch_weight_cache_reload`.
* **Distributed vLLM V1 inference.** Supported models can combine tensor
  parallelism and pipeline parallelism across workers through Xinference's
  xoscar executor, without a separate Ray cluster. See
  :ref:`distributed_inference`.
* **OpenAI-compatible API additions.** A stateless Responses API supports
  clients using ``/v1/responses``. vLLM, SGLang, and MLX also expose reused
  prompt-token counts in usage responses when reported by the engine. See
  :ref:`openai_responses_client` and :ref:`user_guide_client_api`.
* **Unified logging.** The Web UI Log Center provides recent runtime logs
  and, with Elasticsearch configured, historical indexed logs. Request IDs
  connect API, audit, and runtime events. See :ref:`logging` and
  :ref:`user_guide_audit_security`.
* **Installation and local services.** The cross-platform installer and
  service commands simplify setup and local service management on Linux,
  macOS, and Windows. Engine dependencies continue to install on demand in
  per-model virtual environments. See :ref:`one_line_install` and
  :ref:`model_launching_virtualenv`.

If upgrading from a version earlier than 3.0, read :ref:`migration_3_0` for
authentication, Web UI, and container deployment changes. Check each feature
guide for its engine and hardware requirements.

Detailed version notes
----------------------

This page provides a version-by-version index of Xinference release notes.
For detailed updates, please visit the corresponding links below.

+-----------------+--------------------------------------------------------------------------------+
| Version         | Release Notes                                                                  |
+=================+================================================================================+
| v3.5.0          | `View release notes <https://xinference.co/release_notes/v3.5.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v3.4.0          | `View release notes <https://xinference.co/release_notes/v3.4.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v3.3.0          | `View release notes <https://xinference.co/release_notes/v3.3.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v3.2.0          | `View release notes <https://xinference.co/release_notes/v3.2.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v3.1.0          | `View release notes <https://xinference.co/release_notes/v3.1.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v3.0.0          | `View release notes <https://xinference.co/release_notes/v3.0.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.12.0         | `View release notes <https://xinference.co/release_notes/v2.12.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v2.11.0         | `View release notes <https://xinference.co/release_notes/v2.11.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v2.10.0         | `View release notes <https://xinference.co/release_notes/v2.10.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v2.9.0          | `View release notes <https://xinference.co/release_notes/v2.9.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.8.0          | `View release notes <https://xinference.co/release_notes/v2.8.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.7.0          | `View release notes <https://xinference.co/release_notes/v2.7.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.5.0          | `View release notes <https://xinference.co/release_notes/v2.5.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.4.0          | `View release notes <https://xinference.co/release_notes/v2.4.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.3.0          | `View release notes <https://xinference.co/release_notes/v2.3.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.2.0          | `View release notes <https://xinference.co/release_notes/v2.2.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.1.0          | `View release notes <https://xinference.co/release_notes/v2.1.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v2.0.0          | `View release notes <https://xinference.co/release_notes/v2.0.0.html>`_        |
+-----------------+--------------------------------------------------------------------------------+
| v1.17.0         | `View release notes <https://xinference.co/release_notes/v1.17.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v1.16.0         | `View release notes <https://xinference.co/release_notes/v1.16.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v1.15.0         | `View release notes <https://xinference.co/release_notes/v1.15.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v1.14.0         | `View release notes <https://xinference.co/release_notes/v1.14.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v1.13.0         | `View release notes <https://xinference.co/release_notes/v1.13.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v1.12.0         | `View release notes <https://xinference.co/release_notes/v1.12.0.html>`_       |
+-----------------+--------------------------------------------------------------------------------+
| v1.11.0.post1   | `View release notes <https://xinference.co/release_notes/v1.11.0.post1.html>`_ |
+-----------------+--------------------------------------------------------------------------------+
| v1.10.1         | `View release notes <https://xinference.co/release_notes/v1.10.1.html>`_       |
+-----------------+--------------------------------------------------------------------------------+

----

For older versions and source history, see our GitHub releases page:
https://github.com/xorbitsai/inference/releases
