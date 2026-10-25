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

"""Integration tests: SPA routing against a real Next.js static export.

Skipped unless the export has been built. CI builds it in the lint job
(``cd frontend && npm run build``) and runs this module against
``frontend/out``; locally build the export the same way first, or point
``XINFERENCE_TEST_FRONTEND_DIST`` at an export directory.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from xinference.api.frontend_static import mount_frontend

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DIST = Path(
    os.environ.get(
        "XINFERENCE_TEST_FRONTEND_DIST", str(_REPO_ROOT / "frontend" / "out")
    )
)

pytestmark = pytest.mark.skipif(
    not (_DIST / "index.html").is_file(),
    reason="frontend static export not built (run `npm run build` under frontend/)",
)


@pytest.fixture(scope="module")
def client():
    app = FastAPI()

    @app.get("/v1/cluster/auth")
    async def _cluster_auth():
        return JSONResponse({"auth": False})

    assert mount_frontend(app, _DIST) is True
    with TestClient(app) as c:
        yield c


def _assert_html(resp):
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_root_serves_index(client):
    _assert_html(client.get("/"))


@pytest.mark.parametrize(
    "path",
    [
        "/launch-model",
        "/running-model",
        "/register-model",
        "/cluster-information",
        "/login",
    ],
)
def test_static_routes(client, path):
    _assert_html(client.get(path))


@pytest.mark.parametrize(
    "path",
    [
        # launch-model/[modelType]
        "/launch-model/llm",
        # register-model/[modelType]
        "/register-model/llm",
        # register-model/[modelType]/[modelName]
        "/register-model/llm/my-custom-model",
        # running-model/[modelUid]
        "/running-model/qwen3-0",
    ],
)
def test_dynamic_routes_serve_shell(client, path):
    _assert_html(client.get(path))


def test_rsc_prefetch_payload(client):
    resp = client.get("/launch-model.txt")
    assert resp.status_code == 200


def test_next_assets_served(client):
    chunks = list((_DIST / "_next").rglob("*.js"))
    assert chunks, "export has no _next JS chunks"
    rel = chunks[0].relative_to(_DIST).as_posix()
    resp = client.get(f"/{rel}")
    assert resp.status_code == 200


def test_api_route_precedence(client):
    resp = client.get("/v1/cluster/auth")
    assert resp.status_code == 200
    assert resp.json() == {"auth": False}


def test_unmatched_api_prefix_is_json_404(client):
    resp = client.get("/v1/definitely-not-a-route")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_unknown_page_is_404(client):
    resp = client.get("/definitely/not/a/page")
    assert resp.status_code == 404


def test_legacy_ui_redirects(client):
    resp = client.get("/ui/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/"
