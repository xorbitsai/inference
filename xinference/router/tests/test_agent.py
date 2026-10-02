# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from xinference.router.agent import service as agent_service
from xinference.router.agent.control_plane import RouterAgentControlPlaneClient
from xinference.router.agent.process_manager import (
    ManagedRuntimeProcess,
    RouterRuntimeProcessManager,
)
from xinference.router.agent.service import RouterAgent, RouterAgentConfig
from xinference.router.backend import request_headers
from xinference.router.constants import TOKEN_ROUTER_DATA_PLANE_TOKEN_FIELD


class _ControlPlane:
    def __init__(self):
        self.reports = []

    async def report_assignment_status(self, assignment_id, **payload):
        self.reports.append((assignment_id, payload))
        return payload


class _FailingControlPlane(_ControlPlane):
    def __init__(self, failed_state: str):
        super().__init__()
        self.failed_state = failed_state

    async def report_assignment_status(self, assignment_id, **payload):
        await super().report_assignment_status(assignment_id, **payload)
        if payload.get("observed_state") == self.failed_state:
            raise RuntimeError(f"failed to report {self.failed_state}")
        return payload


class _FakeProcess:
    _next_pid = 5000

    def __init__(
        self,
        *,
        returncode: int | None = None,
        ignore_terminate: bool = False,
        terminate_error: OSError | None = None,
        kill_error: OSError | None = None,
    ):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = returncode
        self.ignore_terminate = ignore_terminate
        self.terminate_error = terminate_error
        self.kill_error = kill_error
        self.terminate_calls = 0
        self.kill_calls = 0
        self._done = asyncio.Event()
        if returncode is not None:
            self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if self.terminate_error is not None:
            self.returncode = 0
            self._done.set()
            raise self.terminate_error
        if not self.ignore_terminate:
            self.returncode = 0
            self._done.set()

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9
        self._done.set()
        if self.kill_error is not None:
            raise self.kill_error


def _manager(tmp_path, control=None, **kwargs):
    return RouterRuntimeProcessManager(
        node_id="node-a",
        supervisor_url="http://supervisor:9997",
        internal_token="internal-secret",
        runtime_executable="/test/bin/xinference-router",
        runtime_log_root=str(tmp_path),
        log_level="INFO",
        drain_timeout_seconds=kwargs.pop("drain_timeout_seconds", 0.05),
        max_restart_backoff_seconds=kwargs.pop("max_restart_backoff_seconds", 60),
        control_plane=control or _ControlPlane(),
        **kwargs,
    )


def _agent_config(tmp_path, **overrides) -> RouterAgentConfig:
    runtime = tmp_path / "xinference-router"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    values: dict[str, Any] = {
        "supervisor_url": "http://supervisor:9997",
        "node_id": "node-a",
        "node_host": "127.0.0.1",
        "port_range_start": 12080,
        "port_range_end": 12089,
        "max_instances": 5,
        "runtime_executable": str(runtime),
        "runtime_log_root": str(tmp_path / "runtime-logs"),
        "internal_token": "secret",
        "startup_retry_initial_seconds": 0.001,
        "startup_retry_max_seconds": 0.01,
        "startup_retry_timeout_seconds": 1.0,
    }
    values.update(overrides)
    config = RouterAgentConfig(**values)
    config.validate()
    return config


class _AgentControlPlane:
    def __init__(
        self,
        *,
        register_results=None,
        asset_results=None,
        assignment_results=None,
    ):
        self.register_results = list(register_results or [{}])
        self.asset_results = list(asset_results or [None])
        self.assignment_results = list(assignment_results or [None])
        self.register_calls = 0
        self.asset_calls = 0
        self.assignment_calls = 0
        self.heartbeat_payloads = []
        self.close_calls = 0

    @staticmethod
    def _next(results, default):
        result = results.pop(0) if results else default
        if isinstance(result, BaseException):
            raise result
        return result

    async def register_node(self, payload):
        self.register_calls += 1
        return self._next(self.register_results, {})

    async def watch_asset_bindings(self, node_id, **kwargs):
        self.asset_calls += 1
        return self._next(self.asset_results, None)

    async def watch_assignments(self, node_id, **kwargs):
        self.assignment_calls += 1
        return self._next(self.assignment_results, None)

    async def heartbeat_node(self, node_id, payload):
        self.heartbeat_payloads.append(payload)
        return {"ok": True}

    async def aclose(self):
        self.close_calls += 1


class _AgentProcessManager:
    def __init__(self):
        self.reconciled = []
        self.shutdown_calls = 0

    @property
    def running_count(self):
        return 0

    def observed_assignments(self):
        return []

    async def reconcile(self, assignments):
        self.reconciled.append(list(assignments))

    async def shutdown(self):
        self.shutdown_calls += 1


class _AgentAssetManager:
    def __init__(self):
        self.reconciled = []

    async def reconcile(self, bindings):
        self.reconciled.append(list(bindings))


def _http_status_error(status_code: int, *, retry_after: str | None = None):
    request = httpx.Request("POST", "http://supervisor:9997/register")
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    response = httpx.Response(status_code, request=request, headers=headers)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}", request=request, response=response
    )


def _router_agent(tmp_path, *, control_plane=None, **config_overrides):
    process_manager = _AgentProcessManager()
    asset_manager = _AgentAssetManager()
    agent = RouterAgent(
        _agent_config(tmp_path, **config_overrides),
        control_plane=control_plane or _AgentControlPlane(),
        process_manager=process_manager,
        asset_manager=asset_manager,
    )
    return agent, process_manager, asset_manager


def test_runtime_instance_id_replaces_only_uuid_host_prefix(tmp_path) -> None:
    manager = _manager(tmp_path)
    assignment = _assignment()
    assignment["listen_host"] = "router-agent-1"
    assignment["instance_id"] = "old-host-name-13fdc0e8-34b3-4e7d-ad06-a1db60427557"

    assert manager._runtime_instance_id(assignment) == (
        "router-agent-1-13fdc0e8-34b3-4e7d-ad06-a1db60427557"
    )
    assert manager._runtime_instance_id({**assignment, "instance_id": "legacy"}) == (
        "legacy"
    )


def test_runtime_child_environment_contains_stable_instance_id(tmp_path) -> None:
    manager = _manager(tmp_path)
    assignment = _assignment()
    assignment["instance_id"] = "router-agent-1-13fdc0e8-34b3-4e7d-ad06-a1db60427557"

    env = manager._child_environment(assignment)

    assert env["XINFERENCE_TOKEN_ROUTER_INSTANCE_ID"] == assignment["instance_id"]


def _assignment(generation=1):
    return {
        "assignment_id": "router-a-0",
        "assignment_generation": generation,
        "router_uid": "router-a",
        "listen_host": "127.0.0.1",
        "listen_port": 12080,
        "public_endpoint": "http://127.0.0.1:12080",
        "desired_state": "running",
    }


@pytest.mark.asyncio
async def test_assignment_status_omits_unset_observed_metadata() -> None:
    payloads: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    client = RouterAgentControlPlaneClient("http://supervisor:9997", "internal-secret")
    await client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.report_assignment_status(
            "router-a-0",
            node_id="node-a",
            assignment_generation=1,
            observed_state="draining",
        )
        await client.report_assignment_status(
            "router-a-0",
            node_id="node-a",
            assignment_generation=1,
            observed_state="draining",
            observed={},
        )
    finally:
        await client.aclose()

    assert "observed" not in payloads[0]
    assert payloads[1]["observed"] == {}


def test_agent_config_reads_node_scope_environment(monkeypatch, tmp_path):
    runtime = tmp_path / "xinference-router"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    values = {
        "XINFERENCE_TOKEN_ROUTER_SUPERVISOR_URL": "http://supervisor:9997",
        "XINFERENCE_TOKEN_ROUTER_NODE_ID": "router-node-1",
        "XINFERENCE_TOKEN_ROUTER_NODE_HOST": "127.0.0.1",
        "XINFERENCE_TOKEN_ROUTER_PORT_RANGE_START": "12080",
        "XINFERENCE_TOKEN_ROUTER_PORT_RANGE_END": "12089",
        "XINFERENCE_TOKEN_ROUTER_MAX_INSTANCES": "5",
        "XINFERENCE_TOKEN_ROUTER_RUNTIME_EXECUTABLE": str(runtime),
        "XINFERENCE_TOKEN_ROUTER_RUNTIME_LOG_ROOT": str(tmp_path / "logs"),
        "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN": "secret",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    config = RouterAgentConfig.from_env()

    assert config.node_id == "router-node-1"
    assert config.port_range_start == 12080
    assert config.port_range_end == 12089
    assert config.startup_retry_initial_seconds == 1.0
    assert config.startup_retry_max_seconds == 15.0
    assert config.startup_retry_timeout_seconds == 120.0
    assert not hasattr(config, "router_uid")


def test_agent_config_reads_startup_retry_environment(monkeypatch, tmp_path):
    runtime = tmp_path / "xinference-router"
    runtime.write_text("#!/bin/sh\nexit 0\n")
    runtime.chmod(0o755)
    values = {
        "XINFERENCE_TOKEN_ROUTER_SUPERVISOR_URL": "http://supervisor:9997",
        "XINFERENCE_TOKEN_ROUTER_NODE_ID": "router-node-1",
        "XINFERENCE_TOKEN_ROUTER_NODE_HOST": "127.0.0.1",
        "XINFERENCE_TOKEN_ROUTER_PORT_RANGE_START": "12080",
        "XINFERENCE_TOKEN_ROUTER_PORT_RANGE_END": "12089",
        "XINFERENCE_TOKEN_ROUTER_MAX_INSTANCES": "5",
        "XINFERENCE_TOKEN_ROUTER_RUNTIME_EXECUTABLE": str(runtime),
        "XINFERENCE_TOKEN_ROUTER_RUNTIME_LOG_ROOT": str(tmp_path / "logs"),
        "XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN": "secret",
        "XINFERENCE_TOKEN_ROUTER_AGENT_STARTUP_RETRY_INITIAL_SECONDS": "2",
        "XINFERENCE_TOKEN_ROUTER_AGENT_STARTUP_RETRY_MAX_SECONDS": "20",
        "XINFERENCE_TOKEN_ROUTER_AGENT_STARTUP_RETRY_TIMEOUT_SECONDS": "180",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    config = RouterAgentConfig.from_env()

    assert config.startup_retry_initial_seconds == 2.0
    assert config.startup_retry_max_seconds == 20.0
    assert config.startup_retry_timeout_seconds == 180.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"startup_retry_initial_seconds": 0}, "initial delay"),
        (
            {
                "startup_retry_initial_seconds": 2,
                "startup_retry_max_seconds": 1,
            },
            "maximum",
        ),
        ({"startup_retry_timeout_seconds": 0}, "retry timeout"),
    ],
)
def test_agent_config_rejects_invalid_startup_retry(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _agent_config(tmp_path, **overrides)


@pytest.mark.asyncio
async def test_agent_bootstrap_retries_connect_error(monkeypatch, tmp_path):
    control = _AgentControlPlane(
        register_results=[
            httpx.ConnectError("not ready"),
            httpx.ConnectError("still not ready"),
            {"ok": True},
        ]
    )
    agent, process_manager, asset_manager = _router_agent(
        tmp_path, control_plane=control
    )
    delays = []

    async def record_retry(delay):
        delays.append(delay)

    monkeypatch.setattr(agent, "_wait_for_startup_retry", record_retry)
    monkeypatch.setattr(agent_service.random, "uniform", lambda lower, upper: 1.0)

    assert await agent._bootstrap() is True
    assert control.register_calls == 3
    assert delays == [0.001, 0.002]
    assert process_manager.reconciled == [[]]
    assert asset_manager.reconciled == [[]]


@pytest.mark.asyncio
async def test_agent_bootstrap_retries_temporary_http_status(monkeypatch, tmp_path):
    control = _AgentControlPlane(
        register_results=[_http_status_error(503), {"ok": True}]
    )
    agent, _, _ = _router_agent(tmp_path, control_plane=control)
    delays = []

    async def record_retry(delay):
        delays.append(delay)

    monkeypatch.setattr(agent, "_wait_for_startup_retry", record_retry)
    monkeypatch.setattr(agent_service.random, "uniform", lambda lower, upper: 1.0)

    assert await agent._bootstrap() is True
    assert control.register_calls == 2
    assert delays == [0.001]


@pytest.mark.asyncio
async def test_agent_bootstrap_honors_retry_after(monkeypatch, tmp_path):
    control = _AgentControlPlane(
        register_results=[_http_status_error(429, retry_after="7"), {"ok": True}]
    )
    agent, _, _ = _router_agent(
        tmp_path,
        control_plane=control,
        startup_retry_max_seconds=10,
        startup_retry_timeout_seconds=30,
    )
    delays = []

    async def record_retry(delay):
        delays.append(delay)

    monkeypatch.setattr(agent, "_wait_for_startup_retry", record_retry)

    assert await agent._bootstrap() is True
    assert delays == [7.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
async def test_agent_bootstrap_does_not_retry_permanent_http_status(
    tmp_path, status_code
):
    control = _AgentControlPlane(register_results=[_http_status_error(status_code)])
    agent, process_manager, _ = _router_agent(tmp_path, control_plane=control)

    with pytest.raises(httpx.HTTPStatusError):
        await agent.run()

    assert control.register_calls == 1
    assert control.close_calls == 1
    assert process_manager.shutdown_calls == 1


@pytest.mark.asyncio
async def test_agent_bootstrap_timeout_is_bounded(monkeypatch, tmp_path):
    control = _AgentControlPlane(register_results=[httpx.ConnectError("not ready")])
    agent, process_manager, _ = _router_agent(
        tmp_path,
        control_plane=control,
        startup_retry_timeout_seconds=1,
    )
    clock = 0.0

    def advancing_monotonic():
        nonlocal clock
        clock += 0.4
        return clock

    monkeypatch.setattr(agent_service.time, "monotonic", advancing_monotonic)

    with pytest.raises(agent_service._RouterAgentBootstrapTimeout):
        await agent.run()

    assert control.register_calls == 1
    assert control.close_calls == 1
    assert process_manager.shutdown_calls == 1


@pytest.mark.asyncio
async def test_agent_bootstrap_stop_interrupts_retry(tmp_path):
    control = _AgentControlPlane(register_results=[httpx.ConnectError("not ready")] * 2)
    agent, _, _ = _router_agent(
        tmp_path,
        control_plane=control,
        startup_retry_initial_seconds=30,
        startup_retry_max_seconds=30,
        startup_retry_timeout_seconds=60,
    )

    task = asyncio.create_task(agent._bootstrap())
    for _ in range(20):
        if control.register_calls:
            break
        await asyncio.sleep(0)
    agent.request_stop()

    assert await asyncio.wait_for(task, timeout=1) is False
    assert control.register_calls == 1


@pytest.mark.asyncio
async def test_agent_bootstrap_stop_interrupts_in_flight_operation(tmp_path):
    control = _AgentControlPlane()
    entered = asyncio.Event()

    async def wait_forever(payload):
        control.register_calls += 1
        entered.set()
        await asyncio.Event().wait()

    control.register_node = wait_forever
    agent, _, _ = _router_agent(
        tmp_path,
        control_plane=control,
        startup_retry_timeout_seconds=60,
    )

    task = asyncio.create_task(agent._bootstrap())
    await asyncio.wait_for(entered.wait(), timeout=1)
    agent.request_stop()

    assert await asyncio.wait_for(task, timeout=1) is False
    assert control.register_calls == 1


@pytest.mark.asyncio
async def test_agent_run_starts_after_bootstrap_and_cleans_up(tmp_path):
    control = _AgentControlPlane()
    agent, process_manager, asset_manager = _router_agent(
        tmp_path, control_plane=control
    )
    original_reconcile = process_manager.reconcile

    async def reconcile_and_stop(assignments):
        await original_reconcile(assignments)
        agent.request_stop()

    process_manager.reconcile = reconcile_and_stop

    await agent.run()

    assert process_manager.reconciled == [[]]
    assert asset_manager.reconciled == [[]]
    assert process_manager.shutdown_calls == 1
    assert control.close_calls == 1
    assert [payload["status"] for payload in control.heartbeat_payloads] == ["draining"]


@pytest.mark.asyncio
async def test_agent_bootstrap_retries_initial_snapshot(monkeypatch, tmp_path):
    control = _AgentControlPlane(
        asset_results=[
            httpx.ConnectError("assets not ready"),
            {"cursor": "asset-cursor", "bindings": []},
        ],
        assignment_results=[
            httpx.ReadTimeout("assignments not ready"),
            {"cursor": "assignment-cursor", "assignments": []},
        ],
    )
    agent, _, _ = _router_agent(tmp_path, control_plane=control)

    async def no_wait(delay):
        return None

    monkeypatch.setattr(agent, "_wait_for_startup_retry", no_wait)

    assert await agent._bootstrap() is True
    assert control.asset_calls == 2
    assert control.assignment_calls == 2
    assert agent._asset_cursor == "asset-cursor"
    assert agent._cursor == "assignment-cursor"


@pytest.mark.asyncio
async def test_agent_bootstrap_does_not_retry_invalid_snapshot(tmp_path):
    control = _AgentControlPlane(asset_results=[{"bindings": "invalid"}])
    agent, _, _ = _router_agent(tmp_path, control_plane=control)

    with pytest.raises(ValueError, match="snapshot must be a list"):
        await agent._bootstrap()

    assert control.asset_calls == 1


def test_agent_registration_ignores_removed_legacy_environment(monkeypatch, caplog):
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_NODE_LABELS", "zone=legacy")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSETS", "asset-a")
    monkeypatch.setattr(agent_service.socket, "gethostname", lambda: "router-host")
    monkeypatch.setattr(agent_service, "_software_revision", lambda: "revision-a")
    agent = object.__new__(RouterAgent)
    agent.config = SimpleNamespace(
        node_id="node-a",
        node_host="127.0.0.1",
        port_range_start=12080,
        port_range_end=12089,
        max_instances=5,
    )

    with caplog.at_level(logging.WARNING, logger=agent_service.__name__):
        registration = agent._node_registration()

    assert registration == {
        "node_id": "node-a",
        "advertise_host": "127.0.0.1",
        "port_range_start": 12080,
        "port_range_end": 12089,
        "max_instances": 5,
        "software_version": agent_service.__version__,
        "software_revision": "revision-a",
        "reported_labels": {"system.hostname": "router-host"},
        "capabilities": {"hostname": "router-host"},
    }
    assert "labels" not in registration
    assert "tokenizer_assets" not in registration["capabilities"]
    assert "XINFERENCE_TOKEN_ROUTER_NODE_LABELS" not in caplog.text
    assert "XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSETS" not in caplog.text


def test_agent_gathers_host_cpu_and_memory_resources(monkeypatch):
    memory = SimpleNamespace(used=2048, available=6144, total=8192)
    monkeypatch.setattr(agent_service.psutil, "cpu_percent", lambda: 25.0)
    monkeypatch.setattr(agent_service.psutil, "cpu_count", lambda: 4)
    monkeypatch.setattr(agent_service.psutil, "virtual_memory", lambda: memory)

    assert agent_service._gather_host_resources() == {
        "cpu": {"usage": 0.25, "total": 4},
        "memory": {"used": 2048, "available": 6144, "total": 8192},
    }


def test_agent_host_resource_failure_does_not_break_heartbeat(monkeypatch):
    def _raise():
        raise RuntimeError("metrics unavailable")

    monkeypatch.setattr(agent_service.psutil, "virtual_memory", _raise)

    assert agent_service._gather_host_resources() == {}


def test_agent_heartbeat_payload_contains_host_resources(monkeypatch):
    resources = {
        "cpu": {"usage": 0.5, "total": 8},
        "memory": {"used": 1024, "available": 3072, "total": 4096},
    }
    monkeypatch.setattr(agent_service, "_gather_host_resources", lambda: resources)
    agent = object.__new__(RouterAgent)
    agent.config = SimpleNamespace(max_instances=5)
    agent.process_manager = SimpleNamespace(
        running_count=2,
        observed_assignments=lambda: [{"assignment_id": "router-a-0"}],
    )

    ready = agent._heartbeat_payload("ready")
    draining = agent._heartbeat_payload("draining")

    assert ready["resources"] == resources
    assert ready["available_slots"] == 3
    assert draining["resources"] == resources
    assert draining["available_slots"] == 0


def test_runtime_child_environment_is_allowlisted(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")
    monkeypatch.setenv("XINFERENCE_LOG_FORMAT", "json")
    monkeypatch.setenv("XINFERENCE_LOG_ROTATION", "size")
    monkeypatch.setenv("XINFERENCE_LOG_RETENTION_DAYS", "7")
    monkeypatch.setenv("XINFERENCE_LOG_MAX_BYTES", "4096")
    monkeypatch.setenv("XINFERENCE_LOG_BACKUP_COUNT", "4")
    control = _ControlPlane()
    manager = RouterRuntimeProcessManager(
        node_id="node-a",
        supervisor_url="http://supervisor:9997",
        internal_token="internal-secret",
        runtime_executable="/test/bin/xinference-router",
        runtime_log_root=str(tmp_path),
        log_level="INFO",
        drain_timeout_seconds=1,
        max_restart_backoff_seconds=60,
        control_plane=control,
    )
    assignment = {
        "assignment_id": "router-a-0",
        "assignment_generation": 3,
        "router_uid": "router-a",
    }

    env = manager._child_environment(assignment)

    assert env["PATH"] == "/test/bin"
    assert env["XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN"] == "internal-secret"
    assert env["XINFERENCE_TOKEN_ROUTER_ASSIGNMENT_ID"] == "router-a-0"
    assert env["XINFERENCE_TOKEN_ROUTER_ASSIGNMENT_GENERATION"] == "3"
    assert env["XINFERENCE_TOKEN_ROUTER_NODE_ID"] == "node-a"
    assert env["XINFERENCE_LOG_DIR"] == str(Path(tmp_path) / "router-a-0")
    assert env["XINFERENCE_LOG_ROTATION"] == "size"
    assert env["XINFERENCE_LOG_RETENTION_DAYS"] == "7"
    assert env["XINFERENCE_LOG_MAX_BYTES"] == "4096"
    assert env["XINFERENCE_LOG_BACKUP_COUNT"] == "4"
    assert "UNRELATED_SECRET" not in env


def test_runtime_child_environment_uses_supervisor_data_plane_credential(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN", raising=False)
    manager = _manager(tmp_path)
    assignment = _assignment()
    assignment[TOKEN_ROUTER_DATA_PLANE_TOKEN_FIELD] = "supervisor-data-plane-secret"

    env = manager._child_environment(assignment)

    assert (
        env["XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN"]
        == "supervisor-data-plane-secret"
    )
    assert env["XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN"] == "internal-secret"
    with patch.dict(os.environ, env, clear=True):
        headers = request_headers(
            [(b"authorization", b"Bearer supervisor-data-plane-secret")],
            backend_api_key="",
            request_id="request-a",
        )
    assert "authorization" not in headers


@pytest.mark.asyncio
async def test_runtime_command_uses_argument_array_and_keeps_token_out_of_argv(
    monkeypatch, tmp_path
):
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
    process = _FakeProcess()

    async def create_subprocess_exec(*args: str, **kwargs: Any):
        calls.append((args, kwargs["env"]))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment(generation=3)])

    command, environment = calls[0]
    assert command == (
        "/test/bin/xinference-router",
        "--supervisor-url",
        "http://supervisor:9997",
        "--router-uid",
        "router-a",
        "--host",
        "127.0.0.1",
        "--port",
        "12080",
        "--public-endpoint",
        "http://127.0.0.1:12080",
        "--log-level",
        "INFO",
    )
    assert "internal-secret" not in command
    assert environment["XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN"] == "internal-secret"
    assert environment["XINFERENCE_TOKEN_ROUTER_ASSIGNMENT_GENERATION"] == "3"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_port_conflict_is_reported_without_spawning(monkeypatch, tmp_path):
    control = _ControlPlane()
    manager = _manager(tmp_path, control)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: False)

    async def unexpected_spawn(*args, **kwargs):
        raise AssertionError("Runtime must not be spawned while its port is occupied")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_spawn)
    await manager.reconcile([_assignment()])

    assert control.reports[-1][1]["observed_state"] == "port_conflict"
    assert control.reports[-1][1]["listen_port"] == 12080
    await manager.shutdown()


@pytest.mark.asyncio
async def test_generation_replacement_and_snapshot_orphan_cleanup(
    monkeypatch, tmp_path
):
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    processes = [_FakeProcess(), _FakeProcess()]

    async def create_subprocess_exec(*args, **kwargs):
        return processes.pop(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment(generation=1)])
    first = manager._processes["router-a-0"].process
    await manager.reconcile([_assignment(generation=2)])
    second = manager._processes["router-a-0"].process

    assert first is not None and first.terminate_calls == 1
    assert second is not None and second is not first

    await manager.reconcile([])
    assert second.terminate_calls == 1
    assert manager._processes == {}


@pytest.mark.asyncio
async def test_data_plane_credential_change_restarts_runtime(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    processes = [_FakeProcess(), _FakeProcess()]
    environments: list[dict[str, str]] = []

    async def create_subprocess_exec(*args, **kwargs):
        environments.append(kwargs["env"])
        return processes.pop(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    first_assignment = _assignment()
    first_assignment[TOKEN_ROUTER_DATA_PLANE_TOKEN_FIELD] = "first-hop-secret"
    await manager.reconcile([first_assignment])
    first = manager._processes["router-a-0"].process

    second_assignment = _assignment()
    second_assignment[TOKEN_ROUTER_DATA_PLANE_TOKEN_FIELD] = "second-hop-secret"
    await manager.reconcile([second_assignment])
    second = manager._processes["router-a-0"].process

    assert first is not None and first.terminate_calls == 1
    assert second is not None and second is not first
    assert environments[0]["XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN"] == (
        "first-hop-secret"
    )
    assert environments[1]["XINFERENCE_TOKEN_ROUTER_DATA_PLANE_TOKEN"] == (
        "second-hop-secret"
    )
    await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_state", ["draining", "stopped"])
async def test_shutdown_stops_runtime_when_status_report_fails(
    monkeypatch, tmp_path, failed_state
):
    control = _FailingControlPlane(failed_state)
    manager = _manager(tmp_path, control)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    process = _FakeProcess()

    async def create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    managed = manager._processes["router-a-0"]

    await manager.shutdown()

    assert process.terminate_calls == 1
    assert process.returncode == 0
    assert managed.process is None
    assert manager._processes == {}


@pytest.mark.asyncio
async def test_reconcile_uses_configured_graceful_timeout(monkeypatch, tmp_path):
    manager = _manager(tmp_path, drain_timeout_seconds=37)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    process = _FakeProcess(ignore_terminate=True)

    async def create_subprocess_exec(*args, **kwargs):
        return process

    timeouts = []

    async def capture_timeout(awaitable, timeout):
        timeouts.append(timeout)
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    monkeypatch.setattr(asyncio, "wait_for", capture_timeout)
    await manager.reconcile([])

    assert timeouts == [37]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


@pytest.mark.asyncio
async def test_shutdown_caps_graceful_timeout(monkeypatch, tmp_path):
    manager = _manager(tmp_path, drain_timeout_seconds=7200)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    process = _FakeProcess(ignore_terminate=True)

    async def create_subprocess_exec(*args, **kwargs):
        return process

    timeouts = []

    async def capture_timeout(awaitable, timeout):
        timeouts.append(timeout)
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    monkeypatch.setattr(asyncio, "wait_for", capture_timeout)
    await manager.shutdown()

    assert timeouts == [10.0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


@pytest.mark.asyncio
async def test_shutdown_kills_runtime_after_graceful_timeout(monkeypatch, tmp_path):
    manager = _manager(tmp_path, drain_timeout_seconds=0.01)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    process = _FakeProcess(ignore_terminate=True)

    async def create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    await manager.shutdown()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1


@pytest.mark.asyncio
async def test_shutdown_tolerates_process_exiting_before_terminate(
    monkeypatch, tmp_path
):
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    process = _FakeProcess(terminate_error=ProcessLookupError())

    async def create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    managed = manager._processes["router-a-0"]
    await manager.shutdown()

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert managed.process is None


@pytest.mark.asyncio
async def test_shutdown_tolerates_process_exiting_before_kill(monkeypatch, tmp_path):
    manager = _manager(tmp_path, drain_timeout_seconds=0.01)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    process = _FakeProcess(ignore_terminate=True, kill_error=ProcessLookupError())

    async def create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    managed = manager._processes["router-a-0"]
    await manager.shutdown()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert managed.process is None


@pytest.mark.asyncio
async def test_runtime_restart_uses_backoff_task_independent_of_heartbeat(
    monkeypatch, tmp_path
):
    manager = _manager(tmp_path, max_restart_backoff_seconds=0)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    first = _FakeProcess(returncode=7)
    second = _FakeProcess()
    processes = [first, second]
    spawn_count = 0

    async def create_subprocess_exec(*args, **kwargs):
        nonlocal spawn_count
        spawn_count += 1
        return processes.pop(0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    for _ in range(20):
        if spawn_count == 2:
            break
        await asyncio.sleep(0)

    assert spawn_count == 2
    assert manager._processes["router-a-0"].process is second
    await manager.shutdown()


@pytest.mark.asyncio
async def test_stopping_assignment_cancels_pending_restart(monkeypatch, tmp_path):
    manager = _manager(tmp_path, max_restart_backoff_seconds=60)
    monkeypatch.setattr(manager, "_port_available", lambda host, port: True)
    process = _FakeProcess(returncode=8)

    async def create_subprocess_exec(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    await manager.reconcile([_assignment()])
    for _ in range(20):
        managed = manager._processes["router-a-0"]
        if managed.restart_task is not None:
            break
        await asyncio.sleep(0)
    restart_task = manager._processes["router-a-0"].restart_task

    assert restart_task is not None
    await manager.reconcile([])
    assert restart_task.done()
    assert manager._processes == {}


def test_restart_policy_enters_crash_loop_after_ten_failures(tmp_path):
    manager = _manager(tmp_path)
    managed = ManagedRuntimeProcess(assignment=_assignment())

    states = [manager._schedule_restart(managed) for _ in range(10)]

    assert states[:9] == ["failed"] * 9
    assert states[9] == "crash_loop"


def test_agent_config_rejects_capacity_larger_than_port_pool(monkeypatch, tmp_path):
    runtime = tmp_path / "xinference-router"
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o755)
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_SUPERVISOR_URL", "http://s:9997")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_NODE_ID", "node-a")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_NODE_HOST", "127.0.0.1")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_PORT_RANGE_START", "12080")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_PORT_RANGE_END", "12080")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_MAX_INSTANCES", "2")
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_RUNTIME_EXECUTABLE", str(runtime))
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_RUNTIME_LOG_ROOT", str(tmp_path))
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "secret")

    with pytest.raises(ValueError, match="exceeds its port range"):
        RouterAgentConfig.from_env()


def test_runtime_child_environment_contains_bound_asset(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    asset_path = tmp_path / "asset-a"
    asset_path.mkdir()
    assignment = _assignment(generation=3)
    assignment["tokenizer_asset"] = {
        "asset_id": "asset-a",
        "revision": "v1",
        "fingerprint": "sha256:" + "a" * 64,
        "binding_generation": 7,
        "observed_state": "ready",
        "local_path": str(asset_path),
    }

    env = manager._child_environment(assignment)

    assert env["ROUTER_TOKENIZER_PATH"] == str(asset_path)
    assert env["XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSET_ID"] == "asset-a"
    assert env["XINFERENCE_TOKEN_ROUTER_TOKENIZER_ASSET_REVISION"] == "v1"
    assert env["XINFERENCE_TOKEN_ROUTER_TOKENIZER_BINDING_GENERATION"] == "7"


@pytest.mark.asyncio
async def test_runtime_does_not_start_without_ready_asset(monkeypatch, tmp_path):
    control = _ControlPlane()
    manager = _manager(tmp_path, control=control)
    assignment = _assignment()
    assignment["tokenizer_asset"] = {
        "asset_id": "asset-a",
        "observed_state": "pending",
        "local_path": "",
    }

    async def unexpected_spawn(*args, **kwargs):
        raise AssertionError("Runtime must not be spawned before its Asset is ready")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_spawn)
    await manager.reconcile([assignment])

    assert control.reports[-1][1]["observed_state"] == "failed"
    assert "not ready" in control.reports[-1][1]["last_error"]
    await manager.shutdown()
