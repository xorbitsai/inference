# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Share the synchronous actor RPC loop among connectors in an EngineCore thread."""

import asyncio
import logging
import os
import threading

logger = logging.getLogger(__name__)
_SHUTDOWN_TIMEOUT = 5.0
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

        async def drain():
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await loop.shutdown_asyncgens()

        cleanup = loop.create_task(drain())
        _, unfinished = loop.run_until_complete(
            asyncio.wait({cleanup}, timeout=_SHUTDOWN_TIMEOUT)
        )
        if unfinished:
            logger.warning("Timed out draining Xavier actor loop during shutdown")
            cleanup.cancel()
            # Give cancellation one turn, without waiting on unresponsive peers.
            loop.run_until_complete(asyncio.sleep(0))
        elif not cleanup.cancelled():
            cleanup.result()
    finally:
        try:
            loop.close()
        finally:
            del _local.loop
