.. _user_guide_auth_system:

===================================
Simple OAuth2 System (experimental)
===================================

Xinference builds an In-memory OAuth2 authentication and authorization system using the account-password mode.

.. note::
   Since **v3.0**, Xinference enables the :ref:`advanced, database-backed
   authentication system <user_guide_advanced_auth_system>` by default
   (``XINFERENCE_AUTH_ADVANCED``), replacing the in-memory system described
   below as the default behavior. On first startup, no account exists yet;
   visiting the web UI walks you through a first-run setup page to create
   the initial admin account, backed by the public ``/v1/admin/setup``
   endpoint described in `Initial admin account`_ below.

   The simple, file-based authentication system on this page still works and
   is not going away. To use it instead of the advanced system, set
   ``XINFERENCE_AUTH_ADVANCED=false`` and pass ``--auth-config`` as described
   below. To disable authentication entirely, also set
   ``XINFERENCE_AUTH_ADVANCED=false`` and omit ``--auth-config``.


Permissions
===========
Currently, Xinference system internally defines some interface permissions:

* ``models:list``: Permission to list models and get models' information.
* ``models:read``: Permission to use models.
* ``models:register``: Permission to register custom models.
* ``models:unregister``: Permission to unregister custom models.
* ``models:start``: Permission to launch models.
* ``models:stop``: Permission to stop running models.
* ``admin``: Administrators have permissions for all interfaces.


Startup
=======
All authentication and authorization information needs to be specified and loaded into memory when Xinference is started.
Xinference requires a JSON-formatted file with the following specific fields:

.. code-block:: json

    {
        "auth_config": {
            "algorithm": "HS256",
            "secret_key": "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",
            "token_expire_in_minutes": 30
        },
        "user_config": [
            {
                "username": "user1",
                "password": "secret1",
                "permissions": [
                    "admin"
                ],
                "api_keys": [
                    "sk-72tkvudyGLPMi",
                    "sk-ZOTLIY4gt9w11"
                ]
            },
            {
                "username": "user2",
                "password": "secret2",
                "permissions": [
                    "models:list",
                    "models:read"
                ],
                "api_keys": [
                    "sk-35tkasdyGLYMy",
                    "sk-ALTbgl6ut981w"
                ]
            }
        ]
    }


* ``auth_config``: This field is used to configure security-related information.

   * ``algorithm``: The algorithm used for token generation and parsing. ``HS`` series algorithms are recommended. For example, ``HS256``, ``HS384`` or ``HS512``.

   * ``secret_key``: The secret_key used for token generation and parsing. Use this command to generate the secret_key adapted to the ``HS`` algorithms: ``openssl rand -hex 32``.

   * ``token_expire_in_minutes``: Reserved field indicating the expiration time of the token. The current open-source version of Xinference does not check the expiration time of tokens.

* ``user_config``: This field is used to configure user and permission information. Each user information is composed of these fields:

   * ``username``: string field for username.

   * ``password``: string field for password.

   * ``permissions``: A list containing strings representing the permissions that this user has. The permissions are described as above.

   * ``api_keys``: A list containing strings representing the api-keys of this user. With these api-keys, user can access the xinference interfaces without the need to signin. The api-key here is formatted similar to the ``OPENAI_API_KEY`` , always starting with ``sk-``, followed by 13 alphanumeric characters.


Once you have configured such a JSON file, use the ``--auth-config`` option to enable Xinference with the authentication and authorization system. For example, for local startup:

.. code-block:: bash

   xinference-local -H 0.0.0.0 --auth-config /path/to/your_json_config_file


For distributed startup, just specify this option when starting the supervisor:

.. code-block:: bash

   xinference-supervisor -H <supervisor_ip> --auth-config /path/to/your_json_config_file


Usage
=====
For Xinference with the authentication and authorization system enabled, all usage remains the same, except for the addition of a login step at the beginning or using the api-key.

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


Http Status Code
================
Add the following two HTTP status codes:

* ``401 Unauthorized``: login information or token verifies failed.
* ``403 Forbidden``: No enough permissions when accessing interfaces.

For the command line, SDK, or web UI users, there will be clear information prompts when encountering authorization and permissions issues.


This feature is still in an experimental stage — see `Feedback`_ at the
bottom of this page for how to reach us with issues or suggestions.


.. _user_guide_advanced_auth_system:

=====================================================
Advanced Authentication System (database-backed)
=====================================================

.. versionadded:: 3.0
   The advanced authentication system, and its default-enabled status.

Unlike the simple system above, which loads a static ``auth_config.json``
file into memory, the advanced authentication system stores users,
permissions, API keys, and refresh tokens in a local SQLite database. It
supports creating/managing users and API keys at runtime through REST
endpoints, instead of requiring a server restart to change an ``auth_config.json``
file.

**Since v3.0, this system is enabled by default** — a fresh Xinference
deployment requires authentication out of the box.

Enabling / disabling
=====================
The advanced authentication system is controlled by the
``XINFERENCE_AUTH_ADVANCED`` environment variable:

* Unset, or any of ``1`` / ``true`` / ``yes`` (case-insensitive): enabled (the default).
* Any of ``0`` / ``false`` / ``no`` (case-insensitive): disabled.

It is mutually exclusive with the simple system's ``--auth-config`` option —
passing ``--auth-config`` while ``XINFERENCE_AUTH_ADVANCED`` is (or defaults
to) enabled raises a startup error. To use ``--auth-config`` instead, or to
run with no authentication at all, set ``XINFERENCE_AUTH_ADVANCED=false``:

.. code-block:: bash

   # Fall back to the simple, file-based auth system
   export XINFERENCE_AUTH_ADVANCED=false
   xinference-local -H 0.0.0.0 --auth-config /path/to/your_json_config_file

   # Or disable authentication entirely
   export XINFERENCE_AUTH_ADVANCED=false
   xinference-local -H 0.0.0.0

Initial admin account
=======================
On first startup with the advanced system enabled, the user table is empty
and Xinference does **not** create an admin account automatically. Instead,
two unauthenticated endpoints handle first-run setup:

* ``GET /v1/admin/setup/status``: returns ``{"needs_setup": true, "initialized": false}``
  while no account exists yet.
* ``POST /v1/admin/setup``: creates the first admin account (with all
  permissions) given a ``username`` and ``password``.

.. code-block:: bash

    curl -X POST "<endpoint>/v1/admin/setup" \
      -H "Content-Type: application/json" \
      -d '{"username": "admin", "password": "choose-a-strong-password"}'

The web UI drives this automatically: opening it for the first time
redirects to a setup page that walks you through creating the admin
account, then to the login page.

Both endpoints stay reachable without authentication, but ``/v1/admin/setup``
permanently refuses to create a second account once one exists — the first
successful call wins. Because whoever reaches this endpoint first becomes
the full-privilege administrator, **complete setup immediately after
deploying or upgrading**, before exposing the instance's port to any
untrusted network.

Secrets and storage locations
===============================
The advanced system needs a JWT signing secret and an encryption key (used
to encrypt stored API keys at rest), plus a database file. All three can be
overridden with environment variables; if you don't set them, Xinference
generates and persists them automatically on first run under
``XINFERENCE_HOME`` (``~/.xinference`` by default):

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

Advanced-system permissions
==============================
The advanced system uses the same permission names as the simple system
(``models:list``, ``models:read``, ``models:register``, ``models:unregister``,
``models:start``, ``models:stop``, ``admin``), plus additional scopes for
managing the advanced system itself:

* ``keys:create``: Permission to create API keys (for oneself, or for others when combined with ``keys:manage``).
* ``keys:manage``: Permission to list, update, delete, and reveal any user's API keys.
* ``users:manage``: Permission to create, update, delete users, and manage their permissions.
* ``cache:list`` / ``cache:delete``: Permissions to list/delete cached model files.
* ``virtualenv:list`` / ``virtualenv:delete``: Permissions to list/delete per-model virtual environments.
* ``admin``: Administrators have all of the above.

A caller may only grant permissions they themselves hold — for example, a
user with only ``users:manage`` cannot grant ``admin`` to someone else.

Login and API-key usage
==========================
Login and API-key usage (signin, ``--api-key``, ``Authorization: Bearer``
header, OpenAI-SDK compatibility) work the same way as described in the
`Usage`_ section above for the simple system.

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
