import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from xinference.router.admission import GateSnapshot
from xinference.router.runtime import RouterDisabled, RouterRuntime, RuntimeSnapshot


class FakeTokenization:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.started = False
        self.closed = False
        self.asset_fingerprint = "sha256:measured-fingerprint"
        self.asset_revision = "measured-revision"

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("start failed")
        self.started = True

    async def aclose(self) -> None:
        self.closed = True

    async def snapshot(self):
        return GateSnapshot(active=0, waiting=0, max_active=2, max_queue=8)

    @property
    def worker_pids(self) -> list[int]:
        return [101, 102]


class BlockingTokenization(FakeTokenization):
    def __init__(self) -> None:
        super().__init__()
        self.started_event = asyncio.Event()

    async def start(self) -> None:
        self.started_event.set()
        await asyncio.Event().wait()


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def config(revision: int, *, enabled: bool = True):
    return SimpleNamespace(
        revision=revision,
        enabled=enabled,
        tokenizer_asset_id="deepseek-v4-flash-0731",
        tokenizer_asset_origin="external",
        tokenizer_asset_revision="0731",
        tokenizer_asset_fingerprint="sha256:test-fingerprint",
    )


def snapshot(
    cfg,
    *,
    fail_start: bool = False,
    tokenization: Any | None = None,
) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        config=cfg,
        tokenization=cast(Any, tokenization or FakeTokenization(fail_start=fail_start)),
        policy=cast(Any, SimpleNamespace()),
        gates={},
        client=FakeClient(),
    )


@pytest.mark.asyncio
async def test_apply_drains_old_snapshot_after_inflight_request(monkeypatch) -> None:
    first = config(1)
    second = config(2)
    snapshots = {1: snapshot(first), 2: snapshot(second)}
    monkeypatch.setattr(
        RouterRuntime,
        "_build_snapshot",
        lambda self, cfg: snapshots[cfg.revision],
    )

    runtime = RouterRuntime(first)
    await runtime.start()
    held = await runtime.acquire()

    await runtime.apply(second)

    assert runtime.current is snapshots[2]
    assert held.draining is True
    assert held.closed is False
    assert cast(FakeTokenization, snapshots[2].tokenization).started is True

    await runtime.release(held)

    assert held.closed is True
    assert held.client.closed is True
    assert held.tokenization.closed is True
    await runtime.aclose()


@pytest.mark.asyncio
async def test_cancelled_apply_closes_replacement_snapshot(monkeypatch) -> None:
    first = config(1)
    second = config(2)
    replacement_tokenization = BlockingTokenization()
    first_snapshot = snapshot(first)
    replacement_snapshot = snapshot(second, tokenization=replacement_tokenization)
    snapshots = {1: first_snapshot, 2: replacement_snapshot}
    monkeypatch.setattr(
        RouterRuntime,
        "_build_snapshot",
        lambda self, cfg: snapshots[cfg.revision],
    )

    runtime = RouterRuntime(first)
    await runtime.start()
    apply_task = asyncio.create_task(runtime.apply(second))
    await replacement_tokenization.started_event.wait()

    apply_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await apply_task

    assert runtime.current is first_snapshot
    assert replacement_snapshot.closed is True
    assert replacement_snapshot.client.closed is True
    assert replacement_tokenization.closed is True
    await runtime.aclose()


@pytest.mark.asyncio
async def test_disabled_snapshot_rejects_new_requests(monkeypatch) -> None:
    disabled = config(2, enabled=False)
    disabled_snapshot = snapshot(disabled)
    monkeypatch.setattr(
        RouterRuntime,
        "_build_snapshot",
        lambda self, cfg: disabled_snapshot,
    )

    runtime = RouterRuntime(disabled)
    await runtime.start()

    with pytest.raises(RouterDisabled):
        await runtime.acquire()

    await runtime.aclose()


@pytest.mark.asyncio
async def test_failed_apply_keeps_previous_snapshot(monkeypatch) -> None:
    first = config(1)
    second = config(2)
    first_snapshot = snapshot(first)
    failed_snapshot = snapshot(second, fail_start=True)
    snapshots = {1: first_snapshot, 2: failed_snapshot}
    monkeypatch.setattr(
        RouterRuntime,
        "_build_snapshot",
        lambda self, cfg: snapshots[cfg.revision],
    )

    runtime = RouterRuntime(first)
    await runtime.start()

    with pytest.raises(RuntimeError, match="start failed"):
        await runtime.apply(second)

    assert runtime.current is first_snapshot
    assert first_snapshot.closed is False
    assert failed_snapshot.closed is True
    await runtime.aclose()


@pytest.mark.asyncio
async def test_summary_reports_measured_asset_fingerprint(monkeypatch) -> None:
    runtime_config = config(1)
    runtime_snapshot = snapshot(runtime_config)
    monkeypatch.setattr(
        RouterRuntime,
        "_build_snapshot",
        lambda self, cfg: runtime_snapshot,
    )

    runtime = RouterRuntime(runtime_config)
    await runtime.start()

    summary = await runtime.summary()

    assert summary["tokenizer_asset"]["asset_id"] == "deepseek-v4-flash-0731"
    assert summary["tokenizer_asset"]["revision"] == "measured-revision"
    assert summary["tokenizer_asset"]["fingerprint"] == "sha256:measured-fingerprint"

    await runtime.aclose()
