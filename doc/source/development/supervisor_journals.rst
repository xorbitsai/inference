Supervisor journal storage
==========================

Extensions may implement ``create_supervisor_journal_store(context)`` to select storage for existing Worker ownership and relaunch journals. The context provides ``kind`` (``worker_ownership`` or ``worker_relaunch``), the default file ``path``, the Supervisor actor ``address``, and its optional immutable ``control_session``. Stores implement synchronous ``load(default)`` and ``save(payload)`` using the existing versioned JSON payloads. A save must become durable or raise before returning.

Without a provider, or when it returns ``None``, the existing ``worker-handoffs.json`` and ``worker-relaunches.json`` files remain in use. Multiple providers, invalid stores and provider errors fail explicitly; errors never select a local fallback. Selection follows control-authority initialization. Operators must migrate authoritative records before switching providers. Worker-local execution journals are unchanged.
