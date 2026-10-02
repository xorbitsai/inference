# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

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


def test_cleanup_errors_do_not_interrupt_loop_release():
    loop = acquire_actor_loop()
    reported = []
    loop.set_exception_handler(lambda loop, context: reported.append(context))

    async def listener():
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("listener cleanup failed")

    async def generator():
        try:
            yield
        finally:
            raise RuntimeError("generator cleanup failed")

    async def start():
        gen = generator()
        await gen.__anext__()
        return asyncio.create_task(listener()), gen

    task, gen = loop.run_until_complete(start())
    release_actor_loop(loop)
    assert loop.is_closed()
    assert str(task.exception()) == "listener cleanup failed"
    assert len(reported) == 1
    assert str(reported[0]["exception"]) == "generator cleanup failed"
    assert gen.ag_frame is None


def test_close_failure_clears_loop_reference(monkeypatch):
    from .. import actor_loop

    loop = acquire_actor_loop()
    close = loop.close

    def fail_close():
        raise RuntimeError("close failed")

    monkeypatch.setattr(loop, "close", fail_close)
    try:
        with pytest.raises(RuntimeError, match="close failed"):
            release_actor_loop(loop)
        assert not hasattr(actor_loop._local, "loop")
        release_actor_loop(loop)
        replacement = acquire_actor_loop()
        assert replacement is not loop
        release_actor_loop(replacement)
    finally:
        close()


def test_connector_shutdown_clears_reference_on_failure(connector_module, monkeypatch):
    from types import SimpleNamespace

    connector = SimpleNamespace(_loop=object())
    released = []

    def fail_release(loop):
        released.append(loop)
        raise RuntimeError("release failed")

    monkeypatch.setattr(connector_module, "release_actor_loop", fail_release)
    cls = connector_module.XavierConnector
    with pytest.raises(RuntimeError, match="release failed"):
        cls.shutdown(connector)
    assert connector._loop is None
    cls.shutdown(connector)
    assert len(released) == 1
