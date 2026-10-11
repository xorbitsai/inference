# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the optional fastapi-guard middleware wiring."""

import asyncio
import itertools
import os
import sys
import threading
import time
import unittest

_IP = itertools.count(1)

try:
    import guard  # noqa: F401

    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False


def _unique_ip() -> str:
    n = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


def _restore_env(saved):
    for name in list(os.environ):
        if name.startswith("XINFERENCE_GUARD_"):
            os.environ.pop(name, None)
    os.environ.update(saved)


class TestGuardIntegration(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.pop(k)
            for k in list(os.environ)
            if k.startswith("XINFERENCE_GUARD_")
        }

    def tearDown(self):
        _restore_env(self._saved)

    def _build_app(self):
        from fastapi import FastAPI

        from xinference.api.guard_integration import attach_guard

        app = FastAPI()

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        attach_guard(app)
        return app

    async def _get(self, client_ip):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=self._build_app(), client=(client_ip, 50000))
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return (await client.get("/ping")).status_code

    def test_disabled_by_default(self):
        from fastapi import FastAPI

        from xinference.api.guard_integration import attach_guard

        app = FastAPI()
        attach_guard(app)
        self.assertEqual(len(app.user_middleware), 0)

    def test_fail_loud_without_package(self):
        from fastapi import FastAPI

        from xinference.api.guard_integration import attach_guard

        if _GUARD_AVAILABLE:
            self.skipTest("fastapi-guard is installed")
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        with self.assertRaisesRegex(ImportError, r"xinference\[guard\]"):
            attach_guard(FastAPI())

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_blocked_ip_is_rejected(self):
        blocked = _unique_ip()
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        os.environ["XINFERENCE_GUARD_BLOCKED_IPS"] = blocked
        self.assertEqual(asyncio.run(self._get(blocked)), 403)

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_rate_limit_returns_429(self):
        client_ip = _unique_ip()
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        os.environ["XINFERENCE_GUARD_RATE_LIMIT"] = "2"
        os.environ["XINFERENCE_GUARD_RATE_LIMIT_WINDOW"] = "60"
        codes = [asyncio.run(self._get(client_ip)) for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_passive_mode_never_blocks(self):
        blocked = _unique_ip()
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        os.environ["XINFERENCE_GUARD_PASSIVE_MODE"] = "true"
        os.environ["XINFERENCE_GUARD_BLOCKED_IPS"] = blocked
        self.assertEqual(asyncio.run(self._get(blocked)), 200)

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_redis_stays_off_without_config(self):
        from xinference.api.guard_integration import _build_guard_config

        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        config = _build_guard_config()
        self.assertFalse(config.enable_redis)

    def test_uvicorn_forwarded_allow_ips_follows_guard_trust(self):
        from xinference.api.guard_integration import uvicorn_forwarded_allow_ips

        # Guard off: serve() keeps the historical behavior untouched.
        self.assertEqual(uvicorn_forwarded_allow_ips(), "*")
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        try:
            # Guard on, no proxies listed: loopback only, so a remote peer
            # cannot rewrite its own address.
            self.assertEqual(uvicorn_forwarded_allow_ips(), "127.0.0.1")
            os.environ["XINFERENCE_GUARD_TRUSTED_PROXIES"] = "10.0.0.1, 10.0.0.2"
            self.assertEqual(uvicorn_forwarded_allow_ips(), "10.0.0.1,10.0.0.2")
        finally:
            os.environ.pop("XINFERENCE_GUARD_ENABLED", None)
            os.environ.pop("XINFERENCE_GUARD_TRUSTED_PROXIES", None)

    def test_default_config_disables_body_scan_and_proxy_trust(self):
        import types
        from unittest import mock

        from xinference.api import guard_integration

        captured = {}

        class _StubConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        stub = types.ModuleType("guard")
        stub.SecurityConfig = _StubConfig
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        with mock.patch.dict(sys.modules, {"guard": stub}):
            guard_integration._build_guard_config()
        # Model input is data, not payloads: body signature-scanning must be
        # opt-in, and no proxy is trusted unless explicitly listed.
        self.assertIs(captured["detection_scan_body"], False)
        self.assertEqual(captured["trusted_proxies"], ())
        self.assertIn(
            "microphone=(self)",
            captured["security_headers"]["permissions_policy"],
        )
        self.assertIs(captured["enable_redis"], False)

    @staticmethod
    def _serve_on_ephemeral_port(app, forwarded_allow_ips):
        """Run the app on a real Uvicorn server thread with serve()'s
        proxy-header configuration, and return (server, base_url)."""
        from uvicorn import Config, Server

        config = Config(
            app=app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            proxy_headers=True,
            forwarded_allow_ips=forwarded_allow_ips,
            lifespan="off",
        )
        server = Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        host, port = server.servers[0].sockets[0].getsockname()[:2]
        return server, f"http://{host}:{int(port)}"

    @staticmethod
    def _http_get(url, headers=None):
        from httpx import Client

        with Client() as client:
            return client.get(url, headers=headers or {})

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_untrusted_forwarded_header_cannot_spoof(self):
        """Through the real Uvicorn wrapper: a peer on the blocklist stays
        blocked even when it forges X-Forwarded-For (the connecting peer is
        not a trusted proxy here, so Uvicorn must not rewrite its
        address)."""
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        # The loopback test client must NOT be a trusted proxy for this
        # probe: point the trust list elsewhere.
        os.environ["XINFERENCE_GUARD_TRUSTED_PROXIES"] = "10.99.99.99"
        os.environ["XINFERENCE_GUARD_BLOCKED_IPS"] = "127.0.0.1"

        from xinference.api.guard_integration import uvicorn_forwarded_allow_ips

        server, base_url = self._serve_on_ephemeral_port(
            self._build_app(), uvicorn_forwarded_allow_ips()
        )
        try:
            direct = self._http_get(f"{base_url}/ping")
            forged = self._http_get(
                f"{base_url}/ping", headers={"X-Forwarded-For": "198.51.100.9"}
            )
        finally:
            server.should_exit = True
        self.assertEqual(direct.status_code, 403)
        self.assertEqual(forged.status_code, 403)

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_configured_proxy_forwarded_header_is_honored(self):
        """Loopback listed as a trusted proxy: Uvicorn resolves the client
        from X-Forwarded-For, so the guard sees the forwarded address (a
        blocked forwarded address is rejected; an unlisted one passes)."""
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        os.environ["XINFERENCE_GUARD_TRUSTED_PROXIES"] = "127.0.0.1"
        blocked = _unique_ip()
        os.environ["XINFERENCE_GUARD_BLOCKED_IPS"] = blocked

        from xinference.api.guard_integration import uvicorn_forwarded_allow_ips

        server, base_url = self._serve_on_ephemeral_port(
            self._build_app(), uvicorn_forwarded_allow_ips()
        )
        try:
            via_blocked = self._http_get(
                f"{base_url}/ping", headers={"X-Forwarded-For": blocked}
            )
            via_allowed = self._http_get(
                f"{base_url}/ping", headers={"X-Forwarded-For": _unique_ip()}
            )
        finally:
            server.should_exit = True
        # If the header were ignored, the client would be loopback and both
        # would return 200: the 403 proves the forwarded address was used.
        self.assertEqual(via_blocked.status_code, 403)
        self.assertEqual(via_allowed.status_code, 200)

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_rotating_untrusted_headers_cannot_evade_rate_limit(self):
        """One peer rotating forged forwarded addresses is still one client
        to the per-IP limiter (the loopback peer is not a trusted proxy
        here, so every forged header is ignored)."""
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        os.environ["XINFERENCE_GUARD_TRUSTED_PROXIES"] = "10.99.99.99"
        os.environ["XINFERENCE_GUARD_RATE_LIMIT"] = "2"
        os.environ["XINFERENCE_GUARD_RATE_LIMIT_WINDOW"] = "60"

        from xinference.api.guard_integration import uvicorn_forwarded_allow_ips

        server, base_url = self._serve_on_ephemeral_port(
            self._build_app(), uvicorn_forwarded_allow_ips()
        )
        try:
            codes = [
                self._http_get(
                    f"{base_url}/ping",
                    headers={"X-Forwarded-For": f"198.51.100.{n}"},
                ).status_code
                for n in (10, 11, 12, 13)
            ]
        finally:
            server.should_exit = True
        self.assertEqual(codes, [200, 200, 429, 429])

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_guard_rejection_stays_inside_audit_wrapping(self):
        """serve() attaches guard before registering the audited IP
        middleware, so guard rejections flow back out through the audit
        layer: the request is audited and keeps X-Request-ID."""
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        blocked = _unique_ip()
        os.environ["XINFERENCE_GUARD_BLOCKED_IPS"] = blocked

        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from starlette.middleware.base import BaseHTTPMiddleware

        from xinference.api.guard_integration import attach_guard

        app = FastAPI()

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        attach_guard(app)  # added first -> inner

        audited = []

        class _AuditedIP(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                audited.append(request.url.path)
                response = await call_next(request)
                response.headers["X-Request-ID"] = "req-123"
                return response

        app.add_middleware(_AuditedIP)  # added last -> outer, like serve()

        async def _get():
            transport = ASGITransport(app=app, client=(blocked, 50000))
            async with AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.get("/ping")

        response = asyncio.run(_get())
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.headers["x-request-id"], "req-123")
        self.assertEqual(audited, ["/ping"])

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_inference_bodies_not_scanned_by_default(self):
        """Ordinary code/security conversation input is model data, not
        payloads executed by this API: it passes with the guard on. Body
        signature-scanning is opt-in via XINFERENCE_GUARD_SCAN_BODY."""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from httpx import ASGITransport, AsyncClient

        from xinference.api.guard_integration import attach_guard

        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"
        prompts = [
            "Explain this SQL query: SELECT * FROM users WHERE name = 'Alice';",
            "Show a simple HTML page with <script>alert('hello')</script>.",
            "Review this shell command: cat /etc/passwd",
        ]

        def _make_app():
            app = FastAPI()

            @app.post("/v1/chat/completions")
            async def chat():
                return JSONResponse({"ok": True})

            attach_guard(app)
            return app

        async def _post(content):
            transport = ASGITransport(app=_make_app(), client=(_unique_ip(), 50000))
            async with AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return (
                    await client.post(
                        "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": content}]},
                    )
                ).status_code

        for prompt in prompts:
            self.assertEqual(asyncio.run(_post(prompt)), 200)

        os.environ["XINFERENCE_GUARD_SCAN_BODY"] = "1"
        self.assertEqual(asyncio.run(_post(prompts[0])), 400)

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_permissions_policy_keeps_webui_microphone(self):
        """The default Permissions-Policy must not break the Web UI's
        in-page audio recorder (same-origin getUserMedia audio)."""
        os.environ["XINFERENCE_GUARD_ENABLED"] = "true"

        async def _headers():
            from httpx import ASGITransport, AsyncClient

            transport = ASGITransport(
                app=self._build_app(), client=(_unique_ip(), 50000)
            )
            async with AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return (await client.get("/ping")).headers

        headers = asyncio.run(_headers())
        self.assertEqual(
            headers["permissions-policy"],
            "geolocation=(), microphone=(self), camera=(self)",
        )


if __name__ == "__main__":
    unittest.main()
