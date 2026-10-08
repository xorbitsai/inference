# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import gc
import threading

import pytest

from ..restful import async_restful_client as client_module


@pytest.mark.parametrize(
    "client_class",
    [client_module.AsyncRESTfulModelHandle, client_module.AsyncClient],
)
def test_destructor_without_initialized_owner_loop(client_class, monkeypatch):
    def fail_get_event_loop():
        raise AssertionError("destructor must not request an implicit event loop")

    monkeypatch.setattr(client_module.asyncio, "get_event_loop", fail_get_event_loop)
    client = client_class.__new__(client_class)
    client.session = object()

    client.__del__()

    client.session = None


@pytest.mark.parametrize(
    "client_class",
    [client_module.AsyncRESTfulModelHandle, client_module.AsyncClient],
)
def test_destructor_after_owner_loop_closed(client_class):
    client = client_class.__new__(client_class)
    client.session = object()
    client._session_loop = asyncio.new_event_loop()
    client._session_loop.close()
    client.__del__()
    client.session = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_class",
    [client_module.AsyncRESTfulModelHandle, client_module.AsyncClient],
)
async def test_destructor_schedules_close_on_running_loop(client_class):
    client = client_class.__new__(client_class)
    client._session_loop = asyncio.get_running_loop()
    client.session = object()
    closed = asyncio.Event()

    async def close():
        client.session = None
        closed.set()

    client.close = close

    client.__del__()

    await asyncio.wait_for(closed.wait(), timeout=1)
    assert client.session is None


@pytest.mark.parametrize(
    "client_class",
    [client_module.AsyncRESTfulModelHandle, client_module.AsyncClient],
)
@pytest.mark.parametrize("other_thread", [False, True])
def test_real_session_cleanup_on_paused_owner_loop(
    client_class, other_thread, monkeypatch
):
    monkeypatch.setattr(
        client_module.AsyncClient, "_check_cluster_authenticated", lambda self: None
    )
    loop = asyncio.new_event_loop()
    loop.set_debug(True)

    async def create():
        if client_class is client_module.AsyncClient:
            return client_class("http://localhost:9997")
        return client_class("model", "http://localhost:9997", {})

    clients = [loop.run_until_complete(create())]
    session = clients[0].session
    closed_on = []
    original_close = session.close

    async def close():
        closed_on.append(asyncio.get_running_loop())
        await original_close()

    monkeypatch.setattr(session, "close", close)

    async def drop():
        clients.clear()
        gc.collect()
        await asyncio.sleep(0)

    try:
        if other_thread:
            thread = threading.Thread(target=lambda: asyncio.run(drop()))
            thread.start()
            thread.join()
        else:
            clients.clear()
            gc.collect()
        assert not session.closed
        loop.run_until_complete(asyncio.sleep(0))
        assert session.closed
        assert closed_on == [loop]
    finally:
        loop.run_until_complete(original_close())
        loop.close()
