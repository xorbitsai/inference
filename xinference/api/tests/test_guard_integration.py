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


if __name__ == "__main__":
    unittest.main()
