.. _user_guide_auth_system:

===================================
Simple OAuth2 System (experimental)
===================================

Xinference builds an In-memory OAuth2 authentication and authorization system using the account-password mode.

.. note::
   If you don't have authentication and authorization requirements, you can use Xinference as before, without any changes.


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


Advanced authentication and managed API keys
============================================
The ``--auth-config`` file described above is the legacy authentication mode.
It loads users, permissions, and static API keys from JSON when Xinference
starts. Those API keys can be used for inference, but they are not managed at
runtime.

Advanced authentication is the runtime-managed authentication mode. Enable it
with environment variables instead of ``--auth-config``:

.. code-block:: bash

   export XINFERENCE_AUTH_ADVANCED=true
   export XINFERENCE_AUTH_JWT_SECRET_KEY="$(openssl rand -hex 32)"
   export XINFERENCE_AUTH_ENCRYPTION_KEY="$(openssl rand -hex 32)"
   # Optional. Defaults to ~/.xinference/auth/auth.db.
   export XINFERENCE_AUTH_DB_PATH=/path/to/auth.db
   xinference-local -H 0.0.0.0

``XINFERENCE_AUTH_ADVANCED`` and ``--auth-config`` are mutually exclusive. In
advanced authentication, users, API keys, model permissions, audit information,
and API key usage state are stored in the authentication database. API keys are
created and managed through the web UI or the ``/v1/admin/keys`` management API.

Managed API key controls
------------------------
Managed API keys support these controls:

* ``Token Budget`` limits the total number of tokens an API key may consume.
  When a budget is configured, successful inference responses that include
  usage data count toward the key's usage. This includes OpenAI-compatible
  completion, chat completion, embedding, and rerank responses that report
  ``usage.total_tokens`` or equivalent prompt/input plus completion/output token
  fields. When the used token count reaches the configured budget, later
  inference requests are rejected before model work starts.
* ``Token Renewal`` optionally resets token usage on a schedule. Supported
  schedules are no renewal, daily, monthly, and a custom interval in days.
  Renewal only applies to active, non-expired keys. Expired keys are not renewed
  or re-enabled by the renewal process.
* ``Key Expires After`` sets an expiration timestamp for the key. After the key
  expires, authentication with that key fails with ``401 Unauthorized`` and the
  key cannot be used even if it still has token budget remaining.
* ``Rate Limit`` caps the number of successful requests a key can make in a
  configured time window. This is scoped to the API key, not to the client IP.
  It is separate from the failed-authentication ban protection, which tracks
  invalid, expired, or disabled key attempts and can temporarily ban an IP or an
  IP/key pair.
* ``Rotation`` replaces the secret for an existing API key while preserving the
  key owner, name, description, model permissions, usage-control settings, and
  usage history. The new secret is shown exactly once. The old secret stops
  authenticating immediately after rotation.

The key list in the web UI shows the current state for each key, including used
tokens, remaining token budget, next renewal time, expiration, successful-request
rate-limit state, and last rotation time. It also shows distinct states for
disabled, expired, exhausted-budget, and rate-limited keys.

Create and rotate a key with the web UI
---------------------------------------
In the web UI, open ``API Key Management`` and click ``Create Key``. Choose the
owner, optional expiration, model permissions, and advanced settings such as
token budget, renewal, and request rate limit. After the key is generated, copy
and store the secret immediately. The secret is not shown again from the create
dialog.

To rotate a key, use the rotate action in the key list. Confirm the rotation,
then copy and store the new secret from the one-time display dialog. Existing
clients using the old secret must be updated to use the new secret.

Create and rotate a key with the API
------------------------------------
Log in first and use the returned access token for management requests:

.. code-block:: bash

   export XINFERENCE_ENDPOINT="http://127.0.0.1:9997"

   export ACCESS_TOKEN="$(
     curl -sS -X POST "${XINFERENCE_ENDPOINT}/token" \
       -H "Content-Type: application/json" \
       -d '{"username":"admin","password":"<password>"}' \
       | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
   )"

Create a managed API key:

.. code-block:: bash

   curl -sS -X POST "${XINFERENCE_ENDPOINT}/v1/admin/keys" \
     -H "Authorization: Bearer ${ACCESS_TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "ci pipeline",
       "description": "automation key",
       "expires_at": "2026-12-31T23:59:59",
       "token_budget": 1000000,
       "token_renewal": "monthly",
       "request_rate_limit_enabled": true,
       "request_rate_limit_requests": 120,
       "request_rate_limit_window_seconds": 60,
       "model_permissions": [
         {"permission_type": "all", "permission_value": null}
       ]
     }'

The create response includes the plaintext ``key`` once. Save it immediately.
List and get responses include state such as ``token_usage``,
``token_remaining``, ``token_renewal_next_at``, ``request_rate_limit_count``,
``request_rate_limit_remaining``, ``request_rate_limit_reset_at``, and
``rotated_at``, but do not include the plaintext secret. Existing keys cannot be
revealed again; reveal requests return ``410 Gone``.

Rotate a managed API key:

.. code-block:: bash

   curl -sS -X POST "${XINFERENCE_ENDPOINT}/v1/admin/keys/<key_id>/rotate" \
     -H "Authorization: Bearer ${ACCESS_TOKEN}" \
     -H "Content-Type: application/json" \
     -d '{}'

The rotate response includes the new plaintext ``key`` once, plus the new
``key_prefix`` and ``rotated_at`` timestamp. The old secret is invalid
immediately.


Http Status Code
================
Authentication and authorization errors use the following HTTP status codes:

* ``401 Unauthorized``: login information or token verifies failed.
* ``403 Forbidden``: No enough permissions when accessing interfaces.
* ``429 Too Many Requests``: The request is rejected by token budget exhaustion,
  successful request rate limiting, or failed-authentication ban protection.

For the command line, SDK, or web UI users, there will be clear information prompts when encountering authorization and permissions issues.


Note
====
This feature is still in an experimental stage.
Feel free to provide feedback on usage issues or improvement suggestions through `GitHub issues <https://github.com/xorbitsai/inference/issues>`_ or
`our Telegram group <https://t.me/+nCNpwmySwk9iYmI1>`_.
