from xinference.core.router_config_store import RouterConfigStore


def test_router_config_crud_and_revision(tmp_path):
    store = RouterConfigStore(str(tmp_path / "routers.db"))
    created = store.create("router-a", {"virtual_model_uid": "model-a"}, "admin")
    assert created["revision"] == 1
    assert created["enabled"] is False

    updated = store.update(
        "router-a", {"virtual_model_uid": "model-a", "strategy": "token_budget"}
    )
    assert updated["revision"] == 2
    enabled = store.set_enabled("router-a", True)
    assert enabled["revision"] == 3
    assert enabled["enabled"] is True
    assert store.list()[0]["router_uid"] == "router-a"
    assert store.delete("router-a") is True
    assert store.get("router-a") is None


def test_router_config_rejects_duplicate(tmp_path):
    store = RouterConfigStore(str(tmp_path / "routers.db"))
    store.create("router-a", {"virtual_model_uid": "model-a"})
    try:
        store.create("router-a", {"virtual_model_uid": "model-b"})
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("duplicate router was accepted")


def test_router_config_lookup_by_exact_virtual_model_uid(tmp_path):
    store = RouterConfigStore(str(tmp_path / "routers.db"))
    store.create("router-a", {"virtual_model_uid": "virtual-model"})

    assert store.get_by_virtual_model_uid("virtual-model")["router_uid"] == "router-a"
    assert store.get_by_virtual_model_uid("virtual") is None
    assert store.get_by_virtual_model_uid("virtual-model-suffix") is None
