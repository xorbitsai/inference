.. _migration_3_0:

============================
Migrating to Xinference 3.0
============================

Xinference 3.0 modernizes authentication, the web UI, and container
deployment. Most workloads keep working unchanged, but a few behaviors and
flags were removed or renamed. This page lists every breaking change and how
to adapt.

Authentication is enabled by default
=====================================

Starting with v3.0, the database-backed authentication system is **enabled by
default**. A fresh deployment requires a login or an API key for every API
call.

* On first startup, open the web UI (or call ``POST /v1/admin/setup``) to
  create the initial admin account. See :ref:`user_guide_auth_system` for the
  full flow.
* To restore the previous unauthenticated behavior, set:

  .. code-block:: bash

     export XINFERENCE_AUTH_ADVANCED=false

* Existing clients and scripts that talk to an authenticated server must now
  log in (``xinference login`` / ``client.login()``) or pass an API key
  (``--api-key`` / ``api_key=`` / ``Authorization: Bearer <key>``).

The legacy ``--auth-config`` system is removed
===============================================

The in-memory JSON authentication system configured through the
``--auth-config`` flag of ``xinference-local`` and ``xinference-supervisor``
has been removed, together with its static ``auth_config.json`` format. The
flag no longer exists.

To carry your old users and API keys over to the new database-backed system,
use the migration command:

.. code-block:: bash

   xinference-migrate-auth \
     --auth-config /path/to/old/auth_config.json \
     --db-path ~/.xinference/auth/auth.db \
     --encryption-key <your-encryption-key> \
     --dry-run   # preview first, then run again without --dry-run

The value passed to ``--encryption-key`` must be the **same key** that the
upgraded server uses at runtime: either the value of
``XINFERENCE_AUTH_ENCRYPTION_KEY`` or the value persisted in
``<XINFERENCE_HOME>/auth/encryption_key``. It is not a one-time migration
secret. If a different key is used, the migrated API-key hashes can still
authenticate requests, but Xinference cannot decrypt or reveal the stored
plaintext keys.

If an admin password is ever lost, an operator with shell access can reset it
offline with ``xinference-reset-auth-password`` — see
:ref:`user_guide_auth_system`.

Permission scope renames
=========================

Several fine-grained permission scopes were consolidated and renamed:

.. list-table::
   :header-rows: 1

   * - Legacy scope
     - Replacement
   * - ``models:start``
     - ``models:write``
   * - ``models:stop``
     - ``models:write``
   * - ``models:add``
     - ``models:register``
   * - ``admin`` (for log routes)
     - ``logs:list``
   * - ``admin`` (required by the legacy API-key reveal route)
     - ``keys:manage`` (list any user's keys; update, delete, reveal, and
       manage per-key model permissions)

API key creation remains separate: ``keys:create`` is required to create a
key, and ``keys:manage`` is additionally required when creating one for
another user.

Tokens and API keys that carry the legacy scope names keep working: the
server transparently maps legacy names to their replacements. This
compatibility mapping is deprecated and will be removed in a future release,
so update any automation that grants permissions to use the new names.

New default web UI (Next.js)
=============================

The web UI has been rewritten in Next.js and is now the default; the legacy
React UI has been removed. The UI is exported statically at build time and
served by the Xinference server itself — no separate frontend process is
needed.

* The ``XINFERENCE_FRONTEND_ENDPOINT`` environment variable has been
  **removed**. To serve a custom frontend build, point
  ``XINFERENCE_FRONTEND_DIST_DIR`` at its directory instead.
* The old ``/ui/`` path redirects to ``/``. Opening ``/`` takes you to the
  model launch page.

Built-in Gradio demo pages are removed
=======================================

The per-model Gradio demo UI (previously mounted at ``/{model_uid}``) and the
``/v1/ui/*`` endpoints behind it have been removed, along with the ``gradio``
dependency. To interact with a running model, use the web UI, the
OpenAI-compatible API, or the Python client instead.

The official GPU image requires CUDA 13 and installs engines on demand
=======================================================================

The official 3.0 GPU image is now a slim image based on
``nvidia/cuda:13.0.2-devel-ubuntu22.04``. It requires an NVIDIA driver version
of **580 or later**. The image still includes Python 3.12, the shared CUDA
PyTorch stack, and the Transformers engine, but it no longer pre-installs
vLLM, SGLang, llama.cpp, or other optional inference engines.

When a model needs one of those engines, Xinference installs it into a
per-model virtual environment on first launch. The first launch therefore
takes longer and requires access to PyPI or a compatible private package
mirror; later launches reuse the environment.

For offline or air-gapped Compose deployments, pull or transfer the matching
``xprobe/xinference-pypiserver`` image and pin it to the same release tag as
``xprobe/xinference``. Alternatively, supply your own wheel directory with
the ``docker-compose.byo-wheels.yml`` override. See
:ref:`using_docker_image` and :ref:`using_docker_compose` for the complete
requirements.

Strict Qwen3-family system-message ordering
============================================

For model families whose chat template requires the system message to come
first (including Ornith-1.0-35B, qwen3.5, qwen3.6, and Nex-N2), Xinference now
validates message order before dispatching a request. A ``system`` message at
any position other than ``messages[0]`` returns HTTP 400. Move all system
instructions into a single leading system message before upgrading.

Docker Compose requirements
============================

The bundled Compose files dropped the deprecated ``version:`` field and use
newer Compose features, so deploying with them requires **Docker Compose
v2.24.4 or later**. See :ref:`using_docker_compose`.
