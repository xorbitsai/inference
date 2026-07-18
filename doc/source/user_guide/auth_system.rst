.. _user_guide_auth_system:

=====================================================
Authentication System (database-backed)
=====================================================

.. versionadded:: 3.0
   The database-backed authentication system, enabled by default.

Xinference ships with a database-backed authentication and authorization
system. It stores users, permissions, API keys, and refresh tokens in a
local SQLite database, and supports creating and managing users and API keys
at runtime through REST endpoints — no server restart required.

**Since v3.0, this system is enabled by default** — a fresh Xinference
deployment requires authentication out of the box.

.. note::
   Earlier releases shipped a simpler, in-memory ``--auth-config`` JSON
   system. It has been removed; the database-backed system described here is
   the only authenticated mode.

Enabling / disabling
=====================
Authentication is controlled by the ``XINFERENCE_AUTH_ADVANCED`` environment
variable:

* Unset, or any of ``1`` / ``true`` / ``yes`` (case-insensitive): enabled (the default).
* Any of ``0`` / ``false`` / ``no`` (case-insensitive): disabled — Xinference
  runs with **no authentication at all**, and every endpoint is served
  without a login or API key.

.. code-block:: bash

   # Default: authentication enabled
   xinference-local -H 0.0.0.0

   # Disable authentication entirely
   export XINFERENCE_AUTH_ADVANCED=false
   xinference-local -H 0.0.0.0

Initial admin account
=======================
On first startup with authentication enabled, the user table is empty and
Xinference does **not** create an admin account automatically. Instead, two
unauthenticated endpoints handle first-run setup:

* ``GET /v1/admin/setup/status``: returns ``{"needs_setup": true, "initialized": false}``
  while no account exists yet.
* ``POST /v1/admin/setup``: creates the first admin account (with all
  permissions) given a ``username`` and ``password``.

.. code-block:: bash

    curl -X POST "<endpoint>/v1/admin/setup" \
      -H "Content-Type: application/json" \
      -d '{"username": "admin", "password": "choose-a-strong-password"}'

The web UI drives this automatically: opening it for the first time
redirects to a setup page that asks for the new admin's username and
password, then to the login page.

``/v1/admin/setup`` permanently refuses to create a second account once one
exists — the first successful call wins. On an instance exposed to untrusted
networks before setup completes, whoever reaches this endpoint first becomes
the administrator. If that isn't you, an operator with shell access to the
deployment can take control back with ``xinference-reset-auth-password``
(see `Resetting a lost admin password`_).

Resetting a lost admin password
=================================
If the admin password is lost — or someone else won the first-run setup
race — an operator with shell access to the machine running the RESTful API
can reset an admin's password directly against the auth database, without
logging in:

.. code-block:: bash

    xinference-reset-auth-password --username admin

The command prompts for a new password (or accept ``--password`` on the
command line), updates that admin's password, and revokes the user's active
refresh tokens so any stale sessions stop working. It only operates on users
that already hold the ``admin`` permission, and reads the database at
``XINFERENCE_AUTH_DB_PATH`` (override with ``--db-path``). Run
``xinference-reset-auth-password --help`` for all options.

Secrets and storage locations
===============================
The system needs a JWT signing secret and an encryption key (used to encrypt
stored API keys at rest), plus a database file. All three can be overridden
with environment variables; if you don't set them, Xinference generates and
persists them automatically on first run under ``XINFERENCE_HOME``
(``~/.xinference`` by default):

.. list-table::
   :header-rows: 1

   * - Purpose
     - Environment variable
     - Default location
   * - JWT signing secret
     - ``XINFERENCE_AUTH_JWT_SECRET_KEY``
     - ``<XINFERENCE_HOME>/auth/jwt_secret_key`` (auto-generated)
   * - API key encryption key
     - ``XINFERENCE_AUTH_ENCRYPTION_KEY``
     - ``<XINFERENCE_HOME>/auth/encryption_key`` (auto-generated)
   * - User/API key database
     - ``XINFERENCE_AUTH_DB_PATH``
     - ``<XINFERENCE_HOME>/auth/auth.db``

Auto-generated secrets are written once and reused on subsequent restarts,
so existing JWTs and encrypted API keys keep working across restarts. In a
distributed deployment (supervisor + workers), make sure all processes that
run the RESTful API share the same ``XINFERENCE_HOME`` (or set the same
explicit environment variables), so they agree on the same secrets and
database.

Permissions
===========
Xinference defines the following interface permissions:

* ``models:list``: Permission to list models and get models' information.
* ``models:read``: Permission to use models.
* ``models:write``: Permission to launch and stop models.
* ``models:register``: Permission to register and unregister custom models.
* ``keys:create``: Permission to create API keys (for oneself, or for others when combined with ``keys:manage``).
* ``keys:manage``: Permission to list, update, delete, and reveal any user's API keys.
* ``users:manage``: Permission to create, update, delete users, and manage their permissions.
* ``cache:list`` / ``cache:delete``: Permissions to list/delete cached model files.
* ``virtualenv:list`` / ``virtualenv:delete``: Permissions to list/delete per-model virtual environments.
* ``logs:list``: Permission to view cluster logs.
* ``monitor:view``: Permission to view the monitoring dashboards.
* ``admin``: Administrators have all of the above.

A caller may only grant permissions they themselves hold — for example, a
user with only ``users:manage`` cannot grant ``admin`` to someone else.

.. note::
   Earlier releases used finer-grained scope names: ``models:start`` and
   ``models:stop`` (now ``models:write``), and ``models:add`` /
   ``models:unregister`` (now ``models:register``). Tokens and API keys
   carrying the legacy names keep working — the server transparently maps
   them to the new scopes — but this compatibility mapping is deprecated and
   will be removed in a future release. Use the new names when granting
   permissions.

Usage
=====
With authentication enabled, all usage remains the same, except for the
addition of a login step at the beginning or using an API key.

Signin
------
Signin for command line users:

.. code-block:: bash

   xinference login -e <endpoint> --username <username> --password <password>


For python SDK users:

.. code-block:: python

   from xinference.client import Client
   client = Client('<endpoint>')
   client.login('<name>', '<pass>')


For web UI users, when opening the web UI, you will first be directed to the login page. After logging in, you can use the web UI normally.

Api-Key
-------
For command line users, just add ``--api-key`` or ``-ak`` option in the command you want to use.

.. code-block:: bash

   xinference launch <other options> --api-key <your_api_key>


For python SDK users, pass the ``api_key`` parameter when initializing the client, just like the ``OPENAI`` Python client.

.. code-block:: python

   from xinference.client import Client
   client = Client('<endpoint>', api_key='<your_api_key>')


Xinference is also compatible with the ``OPENAI`` Python SDK as well.

.. code-block:: python

   from openai import OpenAI
   client = OpenAI(base_url="<xinference endpoint>" + "/v1", api_key="<your_api_key>")
   client.models.list()

For http request, pass ``Authorization: Bearer api-key`` in request header.

.. code-block::

    curl --request GET \
      --url "<xinference endpoint>" \
      --header "Authorization: Bearer <your_api_key>"

Managing users and API keys
==============================
Once logged in as a user with the ``users:manage`` and/or ``keys:manage``
permission (the bootstrap ``admin`` account has both), you can manage users
and API keys through REST endpoints under ``/v1/admin``, for example:

.. code-block:: bash

    # Create a new user
    curl -X POST "<endpoint>/v1/admin/users" \
      -H "Authorization: Bearer <admin_access_token>" \
      -H "Content-Type: application/json" \
      -d '{"username": "alice", "password": "s3cret!", "permissions": ["models:list", "models:read"]}'

    # Create an API key for the current user
    curl -X POST "<endpoint>/v1/admin/keys" \
      -H "Authorization: Bearer <access_token>" \
      -H "Content-Type: application/json" \
      -d '{"name": "my-key"}'

Other supported endpoints include listing/updating/deleting users
(``/v1/admin/users``, ``/v1/admin/users/{user_id}``), changing a user's
password (``/v1/admin/users/{user_id}/password``), and listing, updating,
deleting, and revealing API keys (``/v1/admin/keys``,
``/v1/admin/keys/{key_id}``, ``/v1/admin/keys/{key_id}/reveal``).

Managing via the web UI
--------------------------
Everything above is also available from dedicated pages in the web UI:

* **User Management**: create, update, disable, and delete users, and edit
  their permissions (requires ``users:manage``).
* **API Key Management**: create, update, delete, and reveal API keys, with
  optional per-model access restrictions (requires ``keys:create`` for your
  own keys, ``keys:manage`` for other users' keys).
* **Security Settings**: view and tune brute-force protection (login/API-key
  failure rate limits) and unban blocked IPs or keys (requires ``admin``).
  See :ref:`user_guide_audit_security` for details.
* **Audit Center**: browse recorded API activity (requires ``admin``). See
  :ref:`user_guide_audit_security`.

Http Status Code
================
Add the following two HTTP status codes:

* ``401 Unauthorized``: login information or token verifies failed.
* ``403 Forbidden``: No enough permissions when accessing interfaces.

For the command line, SDK, or web UI users, there will be clear information prompts when encountering authorization and permissions issues.

Behavior notes
================

Permission changes take effect without re-login
-------------------------------------------------
Route scope checks read the user's **current** permissions from the
database on every request, not the scopes baked into the JWT at login.
Granting or revoking a permission takes effect on the user's next API
call — no re-login required. This applies to JWT-based browser sessions;
API keys are also live-read (they always have been).

Configurable access-token lifetime
-------------------------------------
The access-token lifetime defaults to 30 minutes and can be overridden
with the ``XINFERENCE_ACCESS_TOKEN_EXPIRE_MINUTES`` environment variable:

.. code-block::

    export XINFERENCE_ACCESS_TOKEN_EXPIRE_MINUTES=10

A shorter lifetime shrinks the token-theft window. The refresh token
lifetime is 7 days and is not currently configurable.

Feedback
==========
This feature is still in an experimental stage.
Feel free to provide feedback on usage issues or improvement suggestions through `GitHub issues <https://github.com/xorbitsai/inference/issues>`_ or
`our Telegram group <https://t.me/+nCNpwmySwk9iYmI1>`_.
