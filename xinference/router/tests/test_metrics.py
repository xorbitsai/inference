from __future__ import annotations

import asyncio

import pytest

from xinference.router.metrics import RouterMetrics


@pytest.mark.asyncio
async def test_runtime_metrics_cover_request_routing_pool_and_process() -> None:
    metrics = RouterMetrics("router-a")

    await metrics.request_started()
    await metrics.assign_request_pool("short")
    await metrics.record_rule_match('rule"one')
    await metrics.record_backend_selection('rule"one', "model-a", "short")
    await metrics.observe_pool_wait("short", 0.02)
    await metrics.record_pool_rejected("short", "queue_full")
    await metrics.observe_backend("model-a", "success", 0.2, pool="short")
    await metrics.finish_request(
        "completed",
        "short",
        duration_seconds=0.3,
        route_mode="non_stream",
    )

    rendered = await metrics.render(
        runtime_summary={
            "revision": 7,
            "pools": {
                "short": {"max_active": 8, "active": 1, "waiting": 2},
            },
        },
        runtime_metadata={
            "router_uid": "router-a",
            "assignment_id": "assignment-a",
            "assignment_generation": 3,
            "software_version": "1.2.3",
            "software_revision": "abc123",
        },
        process={
            "cpu_seconds_total": 12.5,
            "cpu_cores": 0.75,
            "rss_bytes": 1024,
            "virtual_memory_bytes": 4096,
            "thread_count": 5,
            "child_process_count": 2,
            "started_at": 100.0,
            "uptime_seconds": 60.0,
        },
    )

    assert (
        'xinference_token_router_route_requests_total{router_uid="router-a",result="completed",route_mode="non_stream",pool="short"} 1'
        in rendered
    )
    assert (
        'xinference_token_router_requests_total{event="completed",pool="short"} 1'
        in rendered
    )
    assert (
        'xinference_token_router_requests_in_flight{router_uid="router-a",pool="none"} 0'
        in rendered
    )
    assert (
        'xinference_token_router_requests_in_flight{router_uid="router-a",pool="short"} 0'
        in rendered
    )
    assert (
        'xinference_token_router_rule_matches_total{router_uid="router-a",rule_id="rule\\"one",outcome="selected"} 1'
        in rendered
    )
    assert (
        'xinference_token_router_backend_selections_total{router_uid="router-a",backend_model_uid="model-a",rule_id="rule\\"one",pool="short"} 1'
        in rendered
    )
    assert (
        'xinference_token_router_backend_requests_total{router_uid="router-a",backend_model_uid="model-a",result="success",pool="short"} 1'
        in rendered
    )
    assert (
        'xinference_token_router_pool_rejected_total{router_uid="router-a",pool="short",reason="queue_full"} 1'
        in rendered
    )

    histogram_labels = '{router_uid="router-a",result="completed",route_mode="non_stream",pool="short"}'
    assert (
        "xinference_token_router_request_duration_seconds_bucket"
        + histogram_labels[:-1]
        + ',le="0.5"} 1'
        in rendered
    )
    assert (
        "xinference_token_router_request_duration_seconds_bucket"
        + histogram_labels[:-1]
        + ',le="0.25"} 0'
        in rendered
    )
    assert (
        "xinference_token_router_request_duration_seconds_sum"
        + histogram_labels
        + " 0.300000000"
        in rendered
    )
    assert (
        "xinference_token_router_request_duration_seconds_count"
        + histogram_labels
        + " 1"
        in rendered
    )

    assert (
        'xinference_token_router_pool_limit{router_uid="router-a",pool="short"} 8'
        in rendered
    )
    assert (
        'xinference_token_router_build_info{version="1.2.3",commit="abc123"} 1'
        in rendered
    )
    assert (
        'xinference_token_router_config_revision{router_uid="router-a"} 7' in rendered
    )
    assert (
        'xinference_token_router_assignment_generation{router_uid="router-a",assignment_id="assignment-a"} 3'
        in rendered
    )
    assert "xinference_token_router_process_cpu_seconds_total 12.5" in rendered
    assert "xinference_token_router_process_virtual_memory_bytes 4096" in rendered


@pytest.mark.asyncio
async def test_render_is_safe_during_concurrent_updates() -> None:
    metrics = RouterMetrics("router-a")

    async def record(index: int) -> None:
        pool = "short" if index % 2 == 0 else "long"
        await metrics.request_started(pool=pool)
        await metrics.observe_backend("model-a", "success", index / 1000, pool=pool)
        await metrics.finish_request("completed", pool, duration_seconds=index / 1000)

    async def render_repeatedly() -> None:
        for _ in range(20):
            output = await metrics.render()
            assert (
                "# TYPE xinference_token_router_request_duration_seconds histogram"
                in output
            )
            await asyncio.sleep(0)

    await asyncio.gather(*(record(index) for index in range(50)), render_repeatedly())

    rendered = await metrics.render()
    assert rendered.count("xinference_token_router_route_requests_total{") == 2
    assert (
        'xinference_token_router_route_requests_total{router_uid="router-a",result="completed",route_mode="non_stream",pool="short"} 25'
        in rendered
    )
    assert (
        'xinference_token_router_route_requests_total{router_uid="router-a",result="completed",route_mode="non_stream",pool="long"} 25'
        in rendered
    )
