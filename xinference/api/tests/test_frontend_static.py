# Copyright 2022-2026 XProbe Inc.
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

"""Tests for serving the frontend static export (``xinference.api.frontend_static``).

Uses a synthetic export directory rather than a real Next.js build, so the
route-resolution logic can be exercised without a Node toolchain.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from xinference.api.frontend_static import (
    _build_shell_patterns,
    _collect_backend_prefixes,
    ensure_spa_fallback_last,
    mount_frontend,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def dist_dir(tmp_path: Path) -> Path:
    """A synthetic Next.js static export mirroring the real output shape."""
    d = tmp_path / "dist"
    _write(d / "index.html", "<html>index</html>")
    _write(d / "404.html", "<html>not found</html>")
    # Static routes emit both an HTML file and an RSC .txt payload.
    _write(d / "launch-model.html", "<html>launch-model</html>")
    _write(d / "launch-model.txt", "RSC:launch-model")
    _write(d / "login.html", "<html>login</html>")
    _write(d / "login.txt", "RSC:login")
    # Single-segment dynamic route shell.
    _write(d / "running-model" / "__shell__.html", "<html>running-shell</html>")
    _write(d / "running-model" / "__shell__.txt", "RSC:running-shell")
    # Nested double-dynamic route shell (register-model/[modelType]/[modelName]).
    _write(
        d / "register-model" / "__shell__" / "__shell__.html",
        "<html>register-edit-shell</html>",
    )
    _write(
        d / "register-model" / "__shell__" / "__shell__.txt",
        "RSC:register-edit-shell",
    )
    # Hashed build asset.
    _write(d / "_next" / "static" / "chunks" / "main.js", "console.log(1)")
    _write(d / "favicon.ico", "icon-bytes")
    return d


def _app_with_backend_routes(dist: Path) -> FastAPI:
    """An app whose backend routes are registered before the frontend mount."""
    app = FastAPI()

    @app.get("/v1/models/{model_uid}")
    async def _models(model_uid: str):  # pragma: no cover - trivial
        return JSONResponse({"uid": model_uid})

    @app.post("/token")
    async def _token():  # pragma: no cover - trivial
        return JSONResponse({"token": "t"})

    @app.get("/status")
    async def _status():  # pragma: no cover - trivial
        return JSONResponse({"ok": True})

    mounted = mount_frontend(app, dist)
    assert mounted is True
    return app


class TestBuildShellPatterns:
    def test_maps_single_and_double_dynamic_shells(self, dist_dir: Path):
        patterns = _build_shell_patterns(dist_dir)
        assert len(patterns) == 2
        assert any(p.match("running-model/uid-1") for p, _ in patterns)
        assert any(p.match("register-model/llm/my-model") for p, _ in patterns)
        # Static routes must not produce shell patterns.
        assert not any(p.match("launch-model") for p, _ in patterns)

    def test_matches_ids_but_not_extra_segments(self, dist_dir: Path):
        patterns = _build_shell_patterns(dist_dir)
        running = next(p for p, _ in patterns if p.match("running-model/some-uid-1234"))
        assert not running.match("running-model/uid/extra")
        assert not running.match("running-model")


class TestCollectBackendPrefixes:
    def test_derives_prefixes_from_route_table(self, dist_dir: Path):
        app = _app_with_backend_routes(dist_dir)
        prefixes = _collect_backend_prefixes(app)
        assert {"v1", "token", "status"} <= prefixes

    def test_skips_dynamic_first_segments(self, dist_dir: Path):
        app = FastAPI()

        @app.get("/{full_path:path}")
        async def _catchall(full_path: str):  # pragma: no cover - trivial
            return JSONResponse({})

        prefixes = _collect_backend_prefixes(app)
        # The catch-all's dynamic first segment must be skipped; only FastAPI's
        # built-in literal routes (/docs, /redoc, /openapi.json) remain.
        assert not any(p.startswith("{") for p in prefixes)
        assert prefixes == frozenset({"docs", "redoc", "openapi.json"})


class TestMountFrontend:
    def test_not_mounted_when_dist_missing(self, tmp_path: Path):
        app = FastAPI()
        assert mount_frontend(app, tmp_path / "nope") is False

    def test_not_mounted_without_index_html(self, tmp_path: Path):
        d = tmp_path / "dist"
        d.mkdir()
        app = FastAPI()
        assert mount_frontend(app, d) is False


class TestSpaRouting:
    def test_root_serves_index(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "index" in resp.text

    def test_static_route_html(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/launch-model")
        assert resp.status_code == 200
        assert "launch-model" in resp.text

    def test_static_route_rsc_payload(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/launch-model.txt")
        assert resp.status_code == 200
        assert resp.text == "RSC:launch-model"

    def test_static_route_with_trailing_slash(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/launch-model/")
        assert resp.status_code == 200
        assert "launch-model" in resp.text

    def test_dynamic_route_serves_shell(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/running-model/my-model-uid")
        assert resp.status_code == 200
        assert "running-shell" in resp.text

    def test_dynamic_route_with_trailing_slash(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/running-model/my-model-uid/")
        assert resp.status_code == 200
        assert "running-shell" in resp.text

    def test_double_dynamic_route_serves_shell(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/register-model/llm/my-custom-model")
        assert resp.status_code == 200
        assert "register-edit-shell" in resp.text

    def test_dynamic_route_rsc_payload(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/running-model/my-model-uid.txt")
        assert resp.status_code == 200
        assert resp.text == "RSC:running-shell"

    def test_exact_asset(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/favicon.ico")
        assert resp.status_code == 200
        assert resp.text == "icon-bytes"

    def test_next_assets_mounted(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/_next/static/chunks/main.js")
        assert resp.status_code == 200

    def test_api_route_still_wins(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/v1/models/abc")
        assert resp.status_code == 200
        assert resp.json() == {"uid": "abc"}

    def test_unmatched_backend_prefix_is_json_404(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/v1/does-not-exist")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Not Found"}

    def test_unknown_page_serves_404_html(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/definitely/not/a/route")
        assert resp.status_code == 404
        assert "not found" in resp.text

    def test_legacy_ui_path_redirects_to_root(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/ui/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/"
        resp = client.get("/ui", follow_redirects=False)
        assert resp.status_code in (302, 307)

    def test_path_traversal_rejected(self, dist_dir: Path):
        # Write a file outside the dist dir that must never be reachable.
        secret = dist_dir.parent / "secret.txt"
        secret.write_text("top-secret", encoding="utf-8")
        client = TestClient(_app_with_backend_routes(dist_dir))
        resp = client.get("/%2e%2e/secret.txt")
        assert resp.status_code == 404
        assert "top-secret" not in resp.text


class TestEnsureSpaFallbackLast:
    def test_runtime_mount_not_shadowed_after_reorder(self, dist_dir: Path):
        app = _app_with_backend_routes(dist_dir)

        # Simulate a runtime mount (e.g. a per-model Gradio app).
        sub = FastAPI()

        @sub.get("/")
        async def _sub_root():  # pragma: no cover - trivial
            return JSONResponse({"gradio": True})

        app.mount("/my-model-uid", sub)

        client = TestClient(app)
        # Mounted after the catch-all: shadowed until reordered.
        resp = client.get("/my-model-uid/")
        assert resp.status_code == 404

        ensure_spa_fallback_last(app)
        resp = client.get("/my-model-uid/")
        assert resp.status_code == 200
        assert resp.json() == {"gradio": True}

    def test_noop_when_frontend_not_mounted(self):
        app = FastAPI()
        ensure_spa_fallback_last(app)  # must not raise
