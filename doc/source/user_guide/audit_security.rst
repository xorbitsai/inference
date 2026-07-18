.. _user_guide_audit_security:

============================
Audit Logging and Security
============================

When the database-backed authentication system is enabled (the default, see
:ref:`user_guide_auth_system`), Xinference records an audit trail of API
activity and protects the API against brute-force attacks. Administrators
can inspect both from the web UI.

Audit logging
=============

Protected and authenticated API activity is recorded as JSON lines in
``<XINFERENCE_LOG_DIR>/audit.log`` (``<XINFERENCE_HOME>/logs`` by default).
Files rotate daily (and by size) and are kept for
``XINFERENCE_AUDIT_LOG_RETENTION_DAYS`` days (default 90).

This is an authentication and security audit trail, not a complete HTTP
access log. Xinference intentionally excludes ``/v1/audit/*``,
``/v1/cluster/auth``, ``/v1/cluster/ui_config``, ``/status``, and
``/v1/address``; public or bootstrap routes that do not pass through the
authenticated audit path may also be absent.

Each entry contains, among others:

* ``category``: ``inference`` (model calls such as ``/v1/chat/...``,
  ``/v1/embeddings``), ``auth`` (login/token endpoints), or ``admin``
  (everything else).
* ``user``, ``auth_type``, ``api_key_name`` / ``api_key_prefix``: who made
  the call and how it was authenticated.
* ``model_id`` / ``model_name`` / ``model_type``: the model involved, if any.
* ``endpoint``, ``status``, ``latency_ms``, ``client_ip``, ``node``,
  ``address``: what was called, the outcome, and where it ran.

Audit Center
------------

The **Audit Center** page of the web UI lets administrators search and
filter the audit trail (by time range, user, API key, model, category,
status, and client IP). It is backed by ``GET /v1/audit/search``, which
requires the ``admin`` permission.

By default the search reads the local ``audit.log``. If the
``XINFERENCE_ES_URL`` environment variable points at an Elasticsearch
cluster (e.g. with the audit log shipped by Filebeat), the search queries
Elasticsearch instead, using the index pattern from
``XINFERENCE_AUDIT_ES_INDEX`` (default ``xinference-audit-*``).

Brute-force protection
======================

Failed API-key authentication attempts are rate-limited on two levels:

* **Per IP**: an IP presenting invalid API keys is banned after
  ``XINFERENCE_RATE_LIMIT_IP_MAX_FAILURES`` failures (default 10) within
  ``XINFERENCE_RATE_LIMIT_IP_WINDOW_SECONDS`` (default 300), for
  ``XINFERENCE_RATE_LIMIT_IP_BAN_SECONDS`` (default 3600). Requests from a
  banned IP are rejected with ``429 Too Many Requests``.
* **Per (IP, API key)**: repeated failures with one specific key from one IP
  ban that combination after ``XINFERENCE_RATE_LIMIT_KEY_MAX_FAILURES``
  failures (default 5) within
  ``XINFERENCE_RATE_LIMIT_KEY_WINDOW_SECONDS`` (default 300), for
  ``XINFERENCE_RATE_LIMIT_KEY_BAN_SECONDS`` (default 3600).

Individual API keys can override the key-level limits with the
``rate_limit_max_failures`` / ``rate_limit_window_seconds`` /
``rate_limit_ban_seconds`` fields when creating or updating the key.

Security Settings
-----------------

The **Security Settings** page of the web UI lets administrators:

* view and tune the rate-limit configuration at runtime,
* list currently banned IPs and (IP, key) pairs,
* lift bans individually or all at once.

The equivalent REST endpoints live under ``/v1/admin/security/*`` (e.g.
``GET``/``PUT /v1/admin/security/rate-limit``,
``GET /v1/admin/security/banned-ips``,
``POST /v1/admin/security/unban-ip``) and require the ``admin`` permission.
