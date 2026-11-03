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
import http.server
import inspect
import json
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


@pytest.mark.parametrize(
    "client_class",
    [client_module.AsyncRESTfulModelHandle, client_module.AsyncClient],
)
async def test_session_uses_client_timeout(client_class):
    if client_class is client_module.AsyncClient:
        client = client_class("http://localhost:9997")
    else:
        client = client_class("model", "http://localhost:9997", {})
    try:
        # Not aiohttp's 5-minute default, which cuts off long generations.
        assert client.session.timeout.total == 1800
    finally:
        await client.close()


class _FakeCluster:
    """A Xinference REST server on 127.0.0.1 that records every request.

    It runs in its own thread, so a client that blocks the event loop cannot
    block the server it is waiting for: the test fails instead of hanging.
    """

    def __init__(self):
        self.auth = False  # what /v1/cluster/auth reports
        self.probe_failures = []  # statuses of the next /v1/cluster/auth answers
        self.probe_gate = None  # a threading.Event the probe waits for, if set
        self.gate_was_open = []
        self.requests = []  # (path, Authorization header) of every request
        self.logins = []
        cluster = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body):
                data = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:  # the client gave up on this request
                    pass

            def do_GET(self):
                cluster.requests.append((self.path, self.headers.get("Authorization")))
                if self.path != "/v1/cluster/auth":
                    self.reply(200, {"data": []})
                    return
                if cluster.probe_gate is not None:
                    cluster.gate_was_open.append(cluster.probe_gate.wait(timeout=2))
                if cluster.probe_failures:
                    self.reply(cluster.probe_failures.pop(0), {"detail": "boom"})
                else:
                    self.reply(200, {"auth": cluster.auth})

            def do_POST(self):
                cluster.requests.append((self.path, self.headers.get("Authorization")))
                body = self.rfile.read(int(self.headers["Content-Length"]))
                cluster.logins.append(json.loads(body))
                self.reply(200, {"access_token": "session-token"})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def paths(self):
        return [path for path, _ in self.requests]


@pytest.fixture
def cluster():
    fake = _FakeCluster()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


async def test_constructor_sends_no_request(cluster):
    client = client_module.AsyncClient(cluster.url)
    try:
        assert cluster.requests == []
        await client.list_models()
    finally:
        await client.close()

    assert cluster.paths == ["/v1/cluster/auth", "/v1/models"]


async def test_initialisation_does_not_block_the_event_loop(cluster):
    # The probe is answered only after the event loop has run another callback,
    # which it cannot do while the client is blocked inside the probe.
    loop_ran = threading.Event()
    cluster.probe_gate = loop_ran
    asyncio.get_running_loop().call_soon(loop_ran.set)

    client = client_module.AsyncClient(cluster.url)
    try:
        await client.list_models()
    finally:
        await client.close()

    assert cluster.gate_was_open == [True]


async def test_concurrent_first_calls_probe_once_and_send_the_api_key(cluster):
    cluster.auth = True
    client = client_module.AsyncClient(cluster.url, api_key="secret")
    try:
        await asyncio.gather(*(client.list_models() for _ in range(5)))
    finally:
        await client.close()

    assert cluster.paths.count("/v1/cluster/auth") == 1
    assert [auth for path, auth in cluster.requests if path == "/v1/models"] == [
        "Bearer secret"
    ] * 5


@pytest.mark.parametrize(
    "auth, statuses, api_key, authorization",
    [
        pytest.param(True, [], "secret", "Bearer secret", id="auth-on-with-key"),
        pytest.param(True, [], None, None, id="auth-on-without-key"),
        pytest.param(False, [], "secret", None, id="auth-off"),
        # 404: an old server without /v1/cluster/auth
        pytest.param(False, [404], "secret", None, id="no-auth-route"),
    ],
)
async def test_api_key_header_follows_the_cluster_auth(
    cluster, auth, statuses, api_key, authorization
):
    cluster.auth = auth
    cluster.probe_failures = statuses
    client = client_module.AsyncClient(cluster.url, api_key=api_key)
    try:
        await client.list_models()
        await client.list_models()
    finally:
        await client.close()

    # One probe: the outcome is remembered, also when the cluster has no auth.
    assert cluster.requests == [
        ("/v1/cluster/auth", None),
        ("/v1/models", authorization),
        ("/v1/models", authorization),
    ]


async def test_failed_probe_surfaces_on_the_first_call_and_a_retry_succeeds(cluster):
    cluster.auth = True
    cluster.probe_failures = [500]
    client = client_module.AsyncClient(cluster.url, api_key="secret")
    try:
        with pytest.raises(
            RuntimeError, match="Failed to get cluster information, detail: boom"
        ):
            await client.list_models()
        # Nothing was sent without the API key, and the failure is not remembered.
        assert cluster.paths == ["/v1/cluster/auth"]
        assert await client.list_models() == {}
    finally:
        await client.close()

    assert cluster.requests == [
        ("/v1/cluster/auth", None),
        ("/v1/cluster/auth", None),
        ("/v1/models", "Bearer secret"),
    ]


async def test_concurrent_calls_all_see_a_failed_probe(cluster):
    cluster.auth = True
    cluster.probe_failures = [500, 500]
    client = client_module.AsyncClient(cluster.url, api_key="secret")
    try:
        results = await asyncio.gather(
            client.list_models(), client.list_models(), return_exceptions=True
        )
        assert [type(result) for result in results] == [RuntimeError] * 2
        assert "/v1/models" not in cluster.paths
        assert await client.list_models() == {}
    finally:
        await client.close()

    assert cluster.requests[-1] == ("/v1/models", "Bearer secret")


async def test_cancelled_first_call_leaves_the_client_uninitialised(cluster):
    cluster.auth = True
    cluster.probe_gate = threading.Event()  # keeps the first probe waiting
    client = client_module.AsyncClient(cluster.url, api_key="secret")
    try:
        first = asyncio.ensure_future(client.list_models())
        for _ in range(500):
            if cluster.requests:
                break
            await asyncio.sleep(0.01)
        assert cluster.paths == ["/v1/cluster/auth"]
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        cluster.probe_gate.set()
        assert await client.list_models() == {}
    finally:
        await client.close()

    assert cluster.paths == ["/v1/cluster/auth", "/v1/cluster/auth", "/v1/models"]
    assert cluster.requests[-1] == ("/v1/models", "Bearer secret")


@pytest.mark.parametrize(
    "auth, authorization", [(True, "Bearer saved-token"), (False, None)]
)
async def test_token_set_before_the_first_call_follows_the_cluster_auth(
    cluster, auth, authorization
):
    cluster.auth = auth
    client = client_module.AsyncClient(cluster.url, api_key="secret")
    try:
        client._set_token("saved-token")
        await client.list_models()
    finally:
        await client.close()

    assert cluster.requests[-1] == ("/v1/models", authorization)


async def test_login_as_the_first_call_sees_an_auth_enabled_cluster(cluster):
    cluster.auth = True
    client = client_module.AsyncClient(cluster.url, api_key="secret")
    try:
        await client.login("user", "password")
        await client.list_models()
    finally:
        await client.close()

    assert cluster.logins == [{"username": "user", "password": "password"}]
    assert cluster.requests[-1] == ("/v1/models", "Bearer session-token")


@pytest.mark.parametrize(
    "method",
    [
        name
        for name, member in inspect.getmembers(
            client_module.AsyncClient, inspect.iscoroutinefunction
        )
        if name not in {"close", "_check_cluster_authenticated", "_ensure_initialized"}
    ],
)
async def test_every_request_method_initialises_the_client_first(method):
    class Initialised(Exception):
        pass

    async def ensure_initialized():
        raise Initialised

    client = client_module.AsyncClient.__new__(client_module.AsyncClient)
    client.session = None  # a method that skips the init fails without I/O
    client._ensure_initialized = ensure_initialized
    function = getattr(client, method)
    required = [
        "x"
        for parameter in inspect.signature(function).parameters.values()
        if parameter.default is parameter.empty
        and parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]

    with pytest.raises(Initialised):
        await function(*required)
