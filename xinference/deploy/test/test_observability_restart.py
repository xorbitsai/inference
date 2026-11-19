# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""OTEL settings stay authoritative across component replacements."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xinference.core import observability_settings_store as settings
from xinference.core.system_settings_store import SystemSettingsStore
from xinference.deploy.api_host import APIHost


@pytest.fixture
def host(tmp_path, monkeypatch):
    path = str(tmp_path / "system-settings.json")
    store = settings.ObservabilitySettingsStore(path, environ={})
    monkeypatch.setattr(settings, "_store", store)
    host = APIHost(None, {}, SystemSettingsStore(path, environ={}))
    runtime = {"revision": store.get().revision, "enable_otel": False, "error": None}
    host.active = SimpleNamespace(observability=dict(runtime))
    host.supervisor_host = SimpleNamespace(
        active=SimpleNamespace(observability=dict(runtime))
    )
    return host, store


def test_restart_requires_both_components_and_covers_non_toggle_changes(host):
    service, store = host
    assert service.observability_settings("get")["restart_required"] is False
    payload = store.get_public()
    payload["opentelemetry"]["otel_sampling_rate"] = 0.75
    result = service.observability_settings("save", payload)
    assert result["restart_components"] == ["supervisor", "api"]
    service.supervisor_host.active.observability["revision"] = store.get().revision
    assert service.observability_settings("get")["restart_components"] == ["api"]
    service.active.observability["revision"] = store.get().revision
    assert service.observability_settings("get")["restart_required"] is False


def test_restart_keeps_original_defaults_and_never_persists_runtime_metadata(host):
    service, store = host
    baseline = service.observability_settings("baseline")
    payload = store.get_public()
    payload["opentelemetry"]["otlp_base_endpoint"] = "https://collector.example.com"
    service.observability_settings("save", payload)
    service.active.observability["revision"] = store.get().revision
    assert service.observability_settings("baseline") == baseline
    reset = service.observability_settings("reset")
    assert reset["opentelemetry"] == baseline["opentelemetry"]
    assert reset["restart_required"] is True


def test_sdk_failure_and_unhosted_components_do_not_report_applied(host):
    service, _ = host
    service.supervisor_host.active.observability["error"] = "OTEL SDK unavailable"
    result = service.observability_settings("get")
    assert result["restart_required"] is True
    assert "OTEL SDK unavailable" in result["restart_error"]
    service.supervisor_host = None
    assert service.observability_settings("get")["restart_supported"] is False


def test_retired_exporters_shutdown_once_even_after_partial_initialization(monkeypatch):
    from xinference.core import otel

    stopped = []
    providers = [
        SimpleNamespace(shutdown=lambda: stopped.append("trace")),
        SimpleNamespace(shutdown=lambda: stopped.append("metrics")),
    ]
    monkeypatch.setattr(otel, "_otel_providers", providers)
    monkeypatch.setattr(otel, "_otel_initialized", False)
    monkeypatch.setattr(otel, "_otel_shutdown", False)
    otel.shutdown_otel()
    otel.shutdown_otel()
    assert stopped == ["metrics", "trace"]


@pytest.mark.asyncio
async def test_api_sdk_failure_keeps_source_serving(host):
    service, _ = host
    source = service.active
    source.environment = {}
    exited = asyncio.get_running_loop().create_future()
    exited.set_result(0)
    target = SimpleNamespace(
        observability={"error": "OTEL SDK unavailable"},
        process=SimpleNamespace(pid=2, poll=lambda: 0),
        ready=object(),
        wait=AsyncMock(),
        send=AsyncMock(),
        writer=None,
        exited=exited,
    )
    service.spawn = AsyncMock(return_value=target)
    operation = {"python_executable": "python", "requested_executable": None}
    await service.replace(operation)
    assert operation["status"] == "failed"
    assert "OTEL SDK unavailable" in operation["error"]
    assert service.active is source
    target.send.assert_not_called()
