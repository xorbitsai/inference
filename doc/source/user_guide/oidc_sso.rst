.. _user_guide_oidc:

======================
OIDC Single Sign-On
======================

Xinference can authenticate users against an external OpenID Connect (OIDC)
identity provider such as `Keycloak <https://www.keycloak.org/>`_, in
addition to the built-in username/password login. SSO users are provisioned
automatically on first login and managed like any other user afterwards.

OIDC requires the database-backed authentication system to be enabled (the
default) — see :ref:`user_guide_auth_system`.

Configuration
=============

Enable OIDC by setting the following environment variables on the process
that runs the RESTful API:

.. list-table::
   :header-rows: 1

   * - Environment variable
     - Meaning
   * - ``XINFERENCE_OIDC_ENABLED``
     - Set to ``1`` / ``true`` / ``yes`` to enable OIDC. Default: disabled.
   * - ``XINFERENCE_OIDC_ISSUER``
     - Issuer URL of your identity provider, e.g.
       ``https://keycloak.example.com/realms/myrealm``. Xinference discovers
       the provider's endpoints from
       ``<issuer>/.well-known/openid-configuration``.
   * - ``XINFERENCE_OIDC_CLIENT_ID``
     - OAuth client ID registered at the provider.
   * - ``XINFERENCE_OIDC_CLIENT_SECRET``
     - OAuth client secret (a *confidential* client is required).
   * - ``XINFERENCE_OIDC_REDIRECT_URI``
     - Callback URL pointing back at Xinference, i.e.
       ``http(s)://<xinference-endpoint>/api/oidc/callback``. The same URL
       must be registered as a valid redirect URI at the provider.

When ``XINFERENCE_OIDC_ENABLED`` is set, all four remaining variables are
required — Xinference refuses to start and reports the missing ones
otherwise.

Example (Keycloak):

.. code-block:: bash

   export XINFERENCE_OIDC_ENABLED=1
   export XINFERENCE_OIDC_ISSUER=https://keycloak.example.com/realms/myrealm
   export XINFERENCE_OIDC_CLIENT_ID=xinference
   export XINFERENCE_OIDC_CLIENT_SECRET=<client-secret>
   export XINFERENCE_OIDC_REDIRECT_URI=http://xinference.example.com:9997/api/oidc/callback
   xinference-local -H 0.0.0.0

Login flow
==========

* Direct users to ``GET /api/oidc/authorize`` on the Xinference endpoint.
  Xinference redirects the browser to the provider's authorization page
  (requesting the ``openid profile email`` scopes).
* After a successful provider login, the browser returns to
  ``/api/oidc/callback``. Xinference exchanges the authorization code,
  verifies the ID token signature against the provider's published JWKS,
  issues its own access and refresh tokens, and redirects to the web UI.

User provisioning and permissions
==================================

* Users are matched by the OIDC ``sub`` claim. On first login an account is
  created automatically with the provider's ``preferred_username`` (falling
  back to ``email``) as the username and ``source`` set to ``oidc``.
* Newly provisioned SSO users start with only the ``models:list``
  permission. An administrator grants further permissions through the
  **User Management** page or the ``/v1/admin/users`` API — changes take
  effect on the user's next request, without re-login.
* Disabling an SSO user in user management blocks their login even if their
  provider account is still active.
* OIDC users have no local password, so local password login and password
  change do not apply to them.
