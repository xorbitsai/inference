import asyncio
from types import MethodType

import httpx
import pytest
from fastapi import FastAPI

from xinference.api import restful_api as restful_api_module
from xinference.api.restful_api import RESTfulAPI


@pytest.mark.asyncio
async def test_lifespan_starts_and_stops_cluster_metrics(monkeypatch):
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._cluster_metrics_task = None
    api._token_router_client = None
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def _cluster_metrics_update_loop(self):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.set()
            raise

    api._cluster_metrics_update_loop = MethodType(_cluster_metrics_update_loop, api)
    monkeypatch.setattr(restful_api_module, "is_metrics_disabled", lambda: False)
    app = FastAPI(lifespan=api._lifespan)

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=1)
        task = api._cluster_metrics_task
        assert task is not None
        assert task.get_name() == "cluster-metrics-updater"
        assert task.done() is False

    assert stopped.is_set()
    assert task.done()
    assert task.cancelled()
    assert api._cluster_metrics_task is None


@pytest.mark.asyncio
async def test_lifespan_skips_metrics_when_disabled_and_closes_client(monkeypatch):
    api = RESTfulAPI.__new__(RESTfulAPI)
    client = httpx.AsyncClient()
    api._token_router_client = client
    monkeypatch.setattr(restful_api_module, "is_metrics_disabled", lambda: True)
    app = FastAPI(lifespan=api._lifespan)

    async with app.router.lifespan_context(app):
        assert client.is_closed is False
        assert api._cluster_metrics_task is None

    assert client.is_closed is True
    assert api._token_router_client is None
    assert api._cluster_metrics_task is None


@pytest.mark.asyncio
async def test_lifespan_reuses_and_closes_elasticsearch_client(monkeypatch):
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._cluster_metrics_task = None
    api._elasticsearch_client = None
    api._token_router_client = None
    monkeypatch.setattr(restful_api_module, "is_metrics_disabled", lambda: True)
    app = FastAPI(lifespan=api._lifespan)

    async with app.router.lifespan_context(app):
        client = api._get_elasticsearch_client()
        assert api._get_elasticsearch_client() is client
        assert client.closed is False

    assert client.closed is True
    assert api._elasticsearch_client is None


@pytest.mark.asyncio
async def test_lifespan_tolerates_failed_cluster_metrics_task(monkeypatch, caplog):
    api = RESTfulAPI.__new__(RESTfulAPI)
    client = httpx.AsyncClient()
    api._token_router_client = client
    failed = asyncio.Event()

    async def _cluster_metrics_update_loop(self):
        failed.set()
        raise RuntimeError("updater failed")

    api._cluster_metrics_update_loop = MethodType(_cluster_metrics_update_loop, api)
    monkeypatch.setattr(restful_api_module, "is_metrics_disabled", lambda: False)
    app = FastAPI(lifespan=api._lifespan)

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(failed.wait(), timeout=1)
        task = api._cluster_metrics_task
        assert task is not None
        await asyncio.sleep(0)
        assert task.done()

    assert client.is_closed is True
    assert api._token_router_client is None
    assert api._cluster_metrics_task is None
    records = [
        record
        for record in caplog.records
        if record.getMessage() == "Cluster metrics updater failed during shutdown"
    ]
    assert len(records) == 1
    assert records[0].exc_info is not None


@pytest.mark.asyncio
async def test_cluster_metrics_loop_updates_immediately_and_retries(
    monkeypatch, caplog
):
    from xinference.core import metrics as metrics_module

    api = RESTfulAPI.__new__(RESTfulAPI)
    api._supervisor_address = "test-supervisor"
    api._advanced_auth_service = None
    events = []

    class Supervisor:
        def __init__(self):
            self.cluster_calls = 0

        async def get_cluster_metrics_data(self):
            self.cluster_calls += 1
            events.append(f"cluster-{self.cluster_calls}")
            if self.cluster_calls == 1:
                raise RuntimeError("temporary failure")
            return {"cluster": "data"}

        async def list_models(self):
            events.append("models")
            return {"model": "data"}

    supervisor = Supervisor()

    async def _get_supervisor_ref(self):
        events.append("supervisor")
        return supervisor

    async def _sleep(delay):
        events.append(f"sleep-{delay}")
        if events.count(f"sleep-{delay}") == 2:
            raise asyncio.CancelledError

    updates = []

    def _update_cluster_metrics(cluster_data, models_data, supervisor_address):
        events.append("update")
        updates.append((cluster_data, models_data, supervisor_address))

    api._get_supervisor_ref = MethodType(_get_supervisor_ref, api)
    monkeypatch.setattr(restful_api_module.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        metrics_module, "update_cluster_metrics", _update_cluster_metrics
    )

    with pytest.raises(asyncio.CancelledError):
        await api._cluster_metrics_update_loop()

    assert events == [
        "supervisor",
        "cluster-1",
        "sleep-15",
        "supervisor",
        "cluster-2",
        "models",
        "update",
        "sleep-15",
    ]
    assert updates == [({"cluster": "data"}, {"model": "data"}, "test-supervisor")]
    records = [
        record
        for record in caplog.records
        if record.getMessage() == "Failed to update cluster metrics"
    ]
    assert len(records) == 1
    assert records[0].exc_info is not None


@pytest.mark.asyncio
async def test_cluster_metrics_loop_logs_transient_failure_without_traceback(
    monkeypatch, caplog
):
    api = RESTfulAPI.__new__(RESTfulAPI)

    async def _get_supervisor_ref(self):
        raise ConnectionError("supervisor unavailable")

    async def _sleep(delay):
        assert delay == 15
        raise asyncio.CancelledError

    api._get_supervisor_ref = MethodType(_get_supervisor_ref, api)
    monkeypatch.setattr(restful_api_module.asyncio, "sleep", _sleep)

    with pytest.raises(asyncio.CancelledError):
        await api._cluster_metrics_update_loop()

    records = [
        record
        for record in caplog.records
        if record.getMessage()
        == "Failed to update cluster metrics: supervisor unavailable"
    ]
    assert len(records) == 1
    assert records[0].exc_info is None
