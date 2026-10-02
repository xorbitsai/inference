# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Share the synchronous actor RPC loop among connectors in an EngineCore thread."""

import asyncio
import os
import threading

_local = threading.local()


def acquire_actor_loop() -> asyncio.AbstractEventLoop:
    # Never reuse a parent's event loop after fork, or a different thread's loop.
    if getattr(_local, "pid", None) != os.getpid() or not getattr(_local, "users", 0):
        _local.loop = asyncio.new_event_loop()
        _local.pid = os.getpid()
        _local.users = 0
    _local.users += 1
    return _local.loop


def release_actor_loop(loop: asyncio.AbstractEventLoop) -> None:
    if (
        getattr(_local, "pid", None) != os.getpid()
        or getattr(_local, "loop", None) is not loop
    ):
        return
    _local.users -= 1
    if _local.users:
        return
    try:
        # xoscar listener tasks close their clients in cancellation cleanup.
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()
        del _local.loop
