from xinference.core.router_registry import RouterRuntimeRegistry


def test_runtime_registry_lifecycle(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("xinference.core.router_registry.time.time", lambda: now[0])
    registry = RouterRuntimeRegistry(heartbeat_timeout_seconds=10)
    registered = registry.register("router-a", "instance-a", {"endpoint": "x"})
    assert registered["online"] is True
    now[0] = 111.0
    assert registry.get("instance-a")["online"] is False
    heartbeat = registry.heartbeat("instance-a", {"status": "ready"})
    assert heartbeat["online"] is True
    acked = registry.ack("instance-a", 3)
    assert acked["acked_revision"] == 3
    assert registry.unregister("instance-a") is True


def test_runtime_registry_rejects_cross_router_instance_reuse():
    registry = RouterRuntimeRegistry()
    registry.register("router-a", "instance-a", {"acked_revision": 2})

    try:
        registry.register("router-b", "instance-a", {"acked_revision": 0})
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("cross-router instance reuse was accepted")


def test_runtime_registry_rejects_stale_ack():
    registry = RouterRuntimeRegistry()
    registry.register("router-a", "instance-a", {"acked_revision": 2})

    try:
        registry.ack("instance-a", 1)
    except ValueError as exc:
        assert "Stale Token Router ACK" in str(exc)
    else:
        raise AssertionError("stale ACK was accepted")


def test_runtime_registry_purges_instances_after_retention(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("xinference.core.router_registry.time.time", lambda: now[0])
    registry = RouterRuntimeRegistry(
        heartbeat_timeout_seconds=10, stale_retention_seconds=30
    )
    registry.register("router-a", "instance-a", {"endpoint": "x"})

    now[0] = 111.0
    assert registry.list()[0]["online"] is False

    now[0] = 131.0
    assert registry.list() == []
    assert registry.get("instance-a") is None


def test_runtime_registry_rejects_invalid_retention():
    try:
        RouterRuntimeRegistry(heartbeat_timeout_seconds=30, stale_retention_seconds=29)
    except ValueError as exc:
        assert "stale_retention_seconds" in str(exc)
    else:
        raise AssertionError("invalid stale retention was accepted")
