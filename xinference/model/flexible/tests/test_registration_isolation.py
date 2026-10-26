# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace

import pytest


def test_instance_runtime_fields_do_not_change_handoff_catalog(monkeypatch, tmp_path):
    from ....core.worker_handoff import CustomModelState
    from ...custom import RegistryManager
    from .. import core
    from ..custom import register_flexible_model, unregister_flexible_model

    registry = RegistryManager.get_registry("flexible")
    monkeypatch.setattr(registry, "models", [])
    spec = core.FlexibleModelSpec(
        model_name="registration-isolation",
        model_id="registration-isolation",
        model_uri=str(tmp_path),
        launcher="test.launcher",
        launcher_args="{}",
        virtualenv={"packages": ["original-package"]},
    )
    register_flexible_model(spec, persist=False)
    owner = SimpleNamespace(
        _custom_register_type_to_cls={
            "flexible": (
                core.FlexibleModelSpec,
                register_flexible_model,
                unregister_flexible_model,
                None,
            )
        }
    )
    original = CustomModelState.capture(owner)

    def launcher(model_uid, model_spec, **_kwargs):
        return core.FlexibleModel(model_uid, str(tmp_path), model_spec)

    monkeypatch.setattr(core, "get_launcher", lambda _name: launcher)
    first = core.create_flexible_model_instance("first", spec.model_name)
    # Worker launch decorates the instance family with these runtime fields.
    first.model_family.address = "127.0.0.1:12345"
    first.model_family.accelerators = ["0"]
    first.model_family.virtualenv.packages.append("instance-only-package")
    captured = CustomModelState.capture(owner)
    assert captured == original
    second = core.create_flexible_model_instance("second", spec.model_name)
    assert not hasattr(second.model_family, "address")
    assert second.model_family.virtualenv.packages == ["original-package"]
    # A replacement loads the original registration from disk before restore.
    monkeypatch.setattr(
        registry,
        "models",
        [core.FlexibleModelSpec.parse_raw(original["flexible"][spec.model_name])],
    )
    CustomModelState.restore(owner, captured)


@pytest.mark.parametrize("prepared", [False, True])
def test_legacy_snapshot_ignores_placement_but_rejects_changed_definition(
    monkeypatch, tmp_path, prepared
):
    from ....core.worker_handoff import CustomModelState
    from ...custom import RegistryManager
    from .. import core
    from ..custom import register_flexible_model, unregister_flexible_model

    registry = RegistryManager.get_registry("flexible")
    spec = core.FlexibleModelSpec(
        model_name="legacy-placement",
        model_id="legacy-placement",
        model_uri=str(tmp_path),
        launcher="test.launcher",
        launcher_args="{}",
    )
    monkeypatch.setattr(registry, "models", [spec] if prepared else [])
    owner = SimpleNamespace(
        _custom_register_type_to_cls={
            "flexible": (
                core.FlexibleModelSpec,
                register_flexible_model,
                unregister_flexible_model,
                None,
            )
        }
    )
    legacy = spec.copy(deep=True)
    legacy.address = "127.0.0.1:12345"
    legacy.accelerators = ["0"]
    CustomModelState.restore(owner, {"flexible": {spec.model_name: legacy.json()}})
    restored = registry.get_custom_models()[0]
    assert not hasattr(restored, "address")
    assert restored.launcher == spec.launcher
    legacy.launcher_args = '{"changed": true}'
    with pytest.raises(ValueError, match="Custom model definition differs"):
        CustomModelState.restore(owner, {"flexible": {spec.model_name: legacy.json()}})
