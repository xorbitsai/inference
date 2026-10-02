# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..actor_loop import acquire_actor_loop, release_actor_loop


def test_shared_loop_survives_one_owner_and_drains_on_last_release():
    first = acquire_actor_loop()
    second = acquire_actor_loop()
    assert first is second
    cleaned = []

    async def listener():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.append(True)

    async def start():
        return asyncio.create_task(listener())

    task = first.run_until_complete(start())
    release_actor_loop(first)
    assert not second.is_closed()
    assert not task.done()
    release_actor_loop(second)
    assert second.is_closed()
    assert task.cancelled()
    assert cleaned == [True]
    replacement = acquire_actor_loop()
    assert replacement is not first
    release_actor_loop(replacement)


def test_different_threads_have_independent_loops():
    main = acquire_actor_loop()

    def in_thread():
        other = acquire_actor_loop()
        try:
            assert other is not main
        finally:
            release_actor_loop(other)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(in_thread).result()
        assert not main.is_closed()
    finally:
        release_actor_loop(main)


def test_connectors_share_loop_and_shutdown_is_idempotent(connector_module):
    from types import SimpleNamespace

    cls = connector_module.XavierConnector
    first, second = SimpleNamespace(_loop=None), SimpleNamespace(_loop=None)

    async def current():
        return asyncio.get_running_loop()

    assert cls._call(first, current()) is cls._call(second, current())
    cls.shutdown(first)
    cls.shutdown(first)
    assert cls._call(second, current()) is second._loop
    cls.shutdown(second)


def test_alternating_connectors_reuse_router_client(connector_module, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from xoscar.backends.router import Router

    router = Router([], None)
    client = SimpleNamespace(closed=False, dest_address="127.0.0.1:12345")
    create = AsyncMock(return_value=client)
    monkeypatch.setattr(Router, "_create_client", create)
    cls = connector_module.XavierConnector
    connectors = [SimpleNamespace(_loop=None), SimpleNamespace(_loop=None)]
    try:
        for i in range(20):
            assert (
                cls._call(connectors[i % 2], router.get_client("127.0.0.1:12345"))
                is client
            )
        assert create.await_count == 1
    finally:
        for connector in connectors:
            cls.shutdown(connector)
