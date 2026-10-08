import asyncio

import pytest

from xinference.router.admission import AdmissionRejected, CapacityGate


@pytest.mark.asyncio
async def test_queue_full_rejected_and_release_wakes_waiter() -> None:
    gate = CapacityGate(
        "long",
        max_active=1,
        max_queue=1,
        queue_timeout_seconds=1,
        retry_after_seconds=10,
    )
    await gate.acquire()
    waiter = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    with pytest.raises(AdmissionRejected) as exc_info:
        await gate.acquire()
    assert exc_info.value.reason == "queue_full"
    await gate.release()
    await waiter
    snapshot = await gate.snapshot()
    assert snapshot.active == 1
    assert snapshot.waiting == 0
    await gate.release()


@pytest.mark.asyncio
async def test_queue_timeout_does_not_leak_capacity() -> None:
    gate = CapacityGate(
        "short",
        max_active=1,
        max_queue=1,
        queue_timeout_seconds=0.01,
        retry_after_seconds=1,
    )
    await gate.acquire()
    with pytest.raises(AdmissionRejected) as exc_info:
        await gate.acquire()
    assert exc_info.value.reason == "queue_timeout"
    snapshot = await gate.snapshot()
    assert snapshot.active == 1
    assert snapshot.waiting == 0
    await gate.release()
