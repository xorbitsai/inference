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
                ]
            },
            {
                "username": "user2",
                "password": "secret2",
                "permissions": [
                    "models:list",
                    "models:read"
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


Once you have configured such a JSON file, use the ``--auth-config`` option to enable Xinference with the authentication and authorization system. For example, for local startup:

.. code-block:: bash

   xinference-local -H 0.0.0.0 --auth-config /path/to/your_json_config_file


For distributed startup, just specify this option when starting the supervisor:

.. code-block:: bash

   xinference-supervisor -H <supervisor_ip> --auth-config /path/to/your_json_config_file


Usage
=====
For Xinference with the authentication and authorization system enabled, all usage remains the same, except for the addition of a login step at the beginning.

Signin for command line users:

.. code-block:: bash

   xinference login -e <endpoint> --username <username> --password <password>


For python SDK users:

.. code-block:: python

   from xinference.client import Client
   client = Client('<endpoint>')
   client.login('<name>', '<pass>')


For web UI users, when opening the web UI, you will first be directed to the login page. After logging in, you can use the web UI normally.


Http Status Code
================
Add the following two HTTP status codes:

* ``401 Unauthorized``: login information or token verifies failed.
* ``403 Forbidden``: No enough permissions when accessing interfaces.

For the command line, SDK, or web UI users, there will be clear information prompts when encountering authorization and permissions issues.


Note
====
This feature is still in an experimental stage.
Feel free to provide feedback on usage issues or improvement suggestions through `GitHub issues <https://github.com/xorbitsai/inference/issues>`_ or
`our Slack <https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg>`_.
