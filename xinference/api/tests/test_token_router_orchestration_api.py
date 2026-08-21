# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from xinference.api.tests.test_token_router_api import (
    ASSET_ID,
    create_app,
    make_supervisor,
    router_payload,
)


@pytest.mark.asyncio
async def test_managed_router_agent_runtime_lifecycle(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "internal-secret")
    app = create_app(make_supervisor(tmp_path))
    headers = {"Authorization": "Bearer internal-secret"}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/v1/token_routers", json=router_payload())
        assert created.status_code == 201
        revision = created.json()["revision"]

        unauthorized = await client.post(
            "/v1/internal/token-router/nodes/register", json={}
        )
        assert unauthorized.status_code == 401

        registered_node = await client.post(
            "/v1/internal/token-router/nodes/register",
            headers=headers,
            json={
                "node_id": "node-a",
                "advertise_host": "127.0.0.1",
                "port_range_start": 12080,
                "port_range_end": 12089,
                "max_instances": 5,
                "labels": {"zone": "a"},
                "capabilities": {"tokenizer_assets": [ASSET_ID]},
                "software_version": "test",
            },
        )
        assert registered_node.status_code == 200
        assert registered_node.json()["online"] is True

        deployment = await client.put(
            "/v1/token_routers/router-a/deployment",
            json={
                "management_mode": "managed",
                "desired_replicas": 1,
                "placement": {"labels": {"zone": "a"}},
            },
        )
        assert deployment.status_code == 200
        assert deployment.json()["management_mode"] == "managed"

        enabled = await client.post("/v1/token_routers/router-a/enable")
        assert enabled.status_code == 200

        snapshot = await client.get(
            "/v1/internal/token-router/nodes/node-a/assignments",
            headers=headers,
            params={"wait_seconds": 0},
        )
        assert snapshot.status_code == 200
        assert snapshot.json()["full_snapshot"] is True
        assignment = snapshot.json()["assignments"][0]

        starting = await client.put(
            f"/v1/internal/token-router/assignments/{assignment['assignment_id']}/status",
            headers=headers,
            json={
                "node_id": "node-a",
                "assignment_generation": assignment["assignment_generation"],
                "observed_state": "starting",
                "pid": 1234,
                "listen_port": assignment["listen_port"],
            },
        )
        assert starting.status_code == 200

        registration_payload = {
            "router_uid": "router-a",
            "instance_id": "instance-a",
            "endpoint": assignment["public_endpoint"],
            "assignment_id": assignment["assignment_id"],
            "assignment_generation": assignment["assignment_generation"],
            "node_id": "node-a",
            "acked_revision": 0,
        }
        registered_runtime = await client.post(
            "/v1/internal/token-router/instances/register",
            headers=headers,
            json=registration_payload,
        )
        assert registered_runtime.status_code == 200

        heartbeat = await client.post(
            "/v1/internal/token-router/instances/instance-a/heartbeat",
            headers=headers,
            json={"status": "ready", "process": {"pid": 1234}},
        )
        assert heartbeat.status_code == 200

        ack = await client.post(
            "/v1/internal/token-router/instances/instance-a/config-ack",
            headers=headers,
            json={"router_uid": "router-a", "revision": revision},
        )
        assert ack.status_code == 200

        assignments = await client.get("/v1/token_routers/router-a/assignments")
        assert assignments.status_code == 200
        assert assignments.json()[0]["observed_state"] == "ready"
        assert assignments.json()[0]["instance_id"] == "instance-a"

        for instance_id, changes in (
            (
                "stale-generation",
                {"assignment_generation": assignment["assignment_generation"] + 1},
            ),
            ("wrong-node", {"node_id": "node-b"}),
            ("wrong-endpoint", {"endpoint": "http://127.0.0.1:12999"}),
        ):
            rejected = await client.post(
                "/v1/internal/token-router/instances/register",
                headers=headers,
                json={**registration_payload, **changes, "instance_id": instance_id},
            )
            assert rejected.status_code == 409

        external_registration = await client.post(
            "/v1/internal/token-router/instances/register",
            headers=headers,
            json={
                "router_uid": "router-a",
                "instance_id": "missing-assignment",
                "endpoint": assignment["public_endpoint"],
            },
        )
        assert external_registration.status_code == 409

        disabled = await client.post("/v1/token_routers/router-a/disable")
        assert disabled.status_code == 200
        stopping = (await client.get("/v1/token_routers/router-a/assignments")).json()[
            0
        ]
        assert stopping["desired_state"] == "stopped"

        stopped = await client.put(
            f"/v1/internal/token-router/assignments/{assignment['assignment_id']}/status",
            headers=headers,
            json={
                "node_id": "node-a",
                "assignment_generation": assignment["assignment_generation"],
                "observed_state": "stopped",
            },
        )
        assert stopped.status_code == 200
        assert stopped.json()["released"] is True

        deleted = await client.delete("/v1/token_routers/router-a")
        assert deleted.status_code == 200


@pytest.mark.asyncio
async def test_managed_router_enable_creates_binding_for_clean_agent(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "internal-secret")
    app = create_app(make_supervisor(tmp_path))
    headers = {"Authorization": "Bearer internal-secret"}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/v1/token_routers", json=router_payload())
        assert created.status_code == 201
        config = created.json()

        registered = await client.post(
            "/v1/internal/token-router/nodes/register",
            headers=headers,
            json={
                "node_id": "node-a",
                "advertise_host": "127.0.0.1",
                "port_range_start": 12080,
                "port_range_end": 12089,
                "max_instances": 5,
                "labels": {"zone": "a"},
                "capabilities": {},
                "software_version": "test",
            },
        )
        assert registered.status_code == 200

        deployment = await client.put(
            "/v1/token_routers/router-a/deployment",
            json={
                "management_mode": "managed",
                "desired_replicas": 1,
                "placement": {"labels": {"zone": "a"}},
            },
        )
        assert deployment.status_code == 200

        enabled = await client.post("/v1/token_routers/router-a/enable")
        assert enabled.status_code == 200

        bindings = await client.get(f"/v1/tokenizer_assets/{ASSET_ID}/bindings")
        assert bindings.status_code == 200
        assert len(bindings.json()) == 1
        binding = bindings.json()[0]
        assert binding["node_id"] == "node-a"
        assert binding["binding_mode"] == "on_demand"
        assert binding["observed_state"] == "pending"

        assignments = await client.get("/v1/token_routers/router-a/assignments")
        assert assignments.status_code == 200
        assert assignments.json() == []

        ready = await client.post(
            "/v1/internal/token-router/nodes/node-a/asset-bindings/status",
            headers=headers,
            json={
                "asset_id": ASSET_ID,
                "generation": binding["generation"],
                "observed_state": "ready",
                "observed_revision": config["tokenizer_asset_revision"],
                "observed_fingerprint": config["tokenizer_asset_fingerprint"],
                "local_path": f"/assets/{ASSET_ID}",
            },
        )
        assert ready.status_code == 200

        assignments = await client.get("/v1/token_routers/router-a/assignments")
        assert assignments.status_code == 200
        assert len(assignments.json()) == 1
        assert assignments.json()[0]["node_id"] == "node-a"
        assert assignments.json()[0]["tokenizer_asset"]["local_path"] == (
            f"/assets/{ASSET_ID}"
        )


@pytest.mark.asyncio
async def test_external_router_runtime_registration_remains_compatible(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "internal-secret")
    app = create_app(make_supervisor(tmp_path))
    headers = {"Authorization": "Bearer internal-secret"}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.post("/v1/token_routers", json=router_payload())
        ).status_code == 201
        registered = await client.post(
            "/v1/internal/token-router/instances/register",
            headers=headers,
            json={
                "router_uid": "router-a",
                "instance_id": "legacy-external",
                "endpoint": "http://router:10080",
                "acked_revision": 0,
            },
        )

    assert registered.status_code == 200
    assert registered.json()["assignment_id"] is None


@pytest.mark.asyncio
async def test_router_node_active_view_excludes_confirmed_offline_nodes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "internal-secret")
    supervisor = make_supervisor(tmp_path)
    app = create_app(supervisor)
    headers = {"Authorization": "Bearer internal-secret"}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        registered = await client.post(
            "/v1/internal/token-router/nodes/register",
            headers=headers,
            json={
                "node_id": "node-a",
                "advertise_host": "127.0.0.1",
                "port_range_start": 12080,
                "port_range_end": 12089,
                "max_instances": 5,
            },
        )
        assert registered.status_code == 200

        last_seen = (datetime.now(timezone.utc) - timedelta(seconds=46)).isoformat()
        with supervisor._token_router_orchestration.nodes._connect() as conn:
            conn.execute(
                "UPDATE token_router_nodes SET last_seen_at = ? WHERE node_id = ?",
                (last_seen, "node-a"),
            )
        supervisor._token_router_orchestration.sweep_nodes()

        active = await client.get(
            "/v1/token_router_nodes", params={"include_offline": "false"}
        )
        diagnostic = await client.get(
            "/v1/token_router_nodes", params={"include_offline": "true"}
        )
        compatible_default = await client.get("/v1/token_router_nodes")
        invalid = await client.get(
            "/v1/token_router_nodes", params={"include_offline": "sometimes"}
        )

    assert active.status_code == 200
    assert active.json() == []
    assert diagnostic.status_code == 200
    assert diagnostic.json()[0]["connectivity_status"] == "offline"
    assert compatible_default.json()[0]["node_id"] == "node-a"
    assert invalid.status_code == 422
