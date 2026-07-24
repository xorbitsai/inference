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

"""Integration tests for monitor config API endpoints."""

import json
from unittest.mock import MagicMock

import pytest

from xinference.api.routers import admin
from xinference.core.monitor_config_store import MonitorConfigStore


def _json_body(response):
    return json.loads(response.body.decode())


@pytest.fixture
def store(tmp_path):
    return MonitorConfigStore(str(tmp_path / "monitor_config.db"))


@pytest.fixture
def mock_request(store):
    request = MagicMock()
    request.app.state.monitor_config_store = store
    request.app.state.advanced_auth = None
    request.headers = {}
    return request


@pytest.mark.asyncio
async def test_get_ui_config_returns_defaults(mock_request):
    response = await admin.get_ui_config(request=mock_request)
    assert response.status_code == 200
    data = _json_body(response)
    assert data["grafana_url"] == ""
    assert data["grafana_dashboard_uid"] == "xinference-overview"
    assert "grafana_dashboards" in data
    assert data["grafana_dashboards"]["overview"] == "xinference-overview"


@pytest.mark.asyncio
async def test_get_ui_config_after_update(mock_request, store):
    store.update(
        {"grafana_url": "http://grafana.test:3000", "dashboard_overview": "custom-uid"},
        username="admin",
    )
    response = await admin.get_ui_config(request=mock_request)
    data = _json_body(response)
    assert data["grafana_url"] == "http://grafana.test:3000"
    assert data["grafana_dashboard_uid"] == "custom-uid"
    assert data["grafana_dashboards"]["overview"] == "custom-uid"


@pytest.mark.asyncio
async def test_get_monitor_config_returns_sources(mock_request):
    response = await admin.get_monitor_config(request=mock_request)
    assert response.status_code == 200
    data = _json_body(response)
    assert "sources" in data
    assert data["sources"]["grafana_url"] == "default"
    assert "grafana_dashboards" in data


@pytest.mark.asyncio
async def test_update_monitor_config(mock_request):
    body = admin.MonitorConfigUpdate(
        grafana_url="http://new.url",
        cluster_name="test-cluster",
        grafana_dashboards={"overview": "new-overview", "model_load": "new-ml"},
    )
    response = await admin.update_monitor_config(request=mock_request, body=body)
    assert response.status_code == 200
    assert _json_body(response) == {"status": "ok"}

    store = mock_request.app.state.monitor_config_store
    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == "http://new.url"
    assert all_cfg["cluster_name"] == "test-cluster"
    dashboards = store.get_dashboards()
    assert dashboards["overview"] == "new-overview"
    assert dashboards["model_load"] == "new-ml"


@pytest.mark.asyncio
async def test_update_monitor_config_partial(mock_request, store):
    store.update({"grafana_url": "http://first.url"}, username="admin")
    body = admin.MonitorConfigUpdate(cluster_name="new-cluster")
    response = await admin.update_monitor_config(request=mock_request, body=body)
    assert response.status_code == 200

    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == "http://first.url"
    assert all_cfg["cluster_name"] == "new-cluster"


@pytest.mark.asyncio
async def test_check_grafana_empty_url(mock_request):
    body = admin.CheckGrafanaRequest(grafana_url="")
    response = await admin.check_grafana(request=mock_request, body=body)
    assert response.status_code == 400
    data = _json_body(response)
    assert data["ok"] is False
    assert "empty" in data["error"].lower()


@pytest.mark.asyncio
async def test_check_grafana_unreachable(mock_request):
    body = admin.CheckGrafanaRequest(grafana_url="http://192.0.2.1:9999")
    response = await admin.check_grafana(request=mock_request, body=body)
    assert response.status_code == 200
    data = _json_body(response)
    assert data["ok"] is False


@pytest.mark.asyncio
async def test_reset_monitor_config(mock_request, store):
    store.update({"grafana_url": "http://db.url"}, username="admin")
    response = await admin.reset_monitor_config(request=mock_request)
    assert response.status_code == 200
    assert _json_body(response) == {"status": "ok"}

    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == ""
    sources = store.get_sources()
    assert sources["grafana_url"] == "default"


@pytest.mark.asyncio
async def test_full_flow(mock_request, store):
    # 1. Initial state: defaults
    response = await admin.get_ui_config(request=mock_request)
    data = _json_body(response)
    assert data["grafana_url"] == ""

    # 2. Update via API
    body = admin.MonitorConfigUpdate(grafana_url="http://updated.url")
    response = await admin.update_monitor_config(request=mock_request, body=body)
    assert response.status_code == 200

    # 3. Verify update
    response = await admin.get_ui_config(request=mock_request)
    data = _json_body(response)
    assert data["grafana_url"] == "http://updated.url"

    # 4. Reset
    response = await admin.reset_monitor_config(request=mock_request)
    assert response.status_code == 200

    # 5. Verify reset falls back to defaults
    response = await admin.get_ui_config(request=mock_request)
    data = _json_body(response)
    assert data["grafana_url"] == ""


# All six dashboard tab keys, matching MONITOR_DASHBOARD_TABS in the frontend.
ALL_DASHBOARD_KEYS = ("overview", "model_load", "llm_slo", "gpu", "host", "security")


@pytest.mark.asyncio
async def test_get_monitor_config_returns_configured_keys(mock_request):
    response = await admin.get_monitor_config(request=mock_request)
    assert response.status_code == 200
    data = _json_body(response)
    # Fresh store: only overview is configured (always-on default). The other
    # five resolve to code-default UIDs but must NOT be reported as configured.
    assert data["grafana_dashboards_configured"] == ["overview"]


@pytest.mark.asyncio
async def test_put_unchanged_dashboards_does_not_expand_configured(mock_request):
    """Regression for the GET->PUT round-trip. The config dialog submits all
    six dashboard keys: the resolved UID for enabled tabs and an empty string
    for disabled ones. An unchanged save must not persist fallback UIDs for
    disabled tabs, otherwise every tab would become "configured" and the
    backward-compatible single-dashboard UX would break."""
    get_resp = await admin.get_monitor_config(request=mock_request)
    dashboards = _json_body(get_resp)["grafana_dashboards"]

    # Mirror the dialog payload: enabled tab (overview) keeps its UID, rest empty.
    payload = {key: "" for key in ALL_DASHBOARD_KEYS}
    payload["overview"] = dashboards["overview"]
    body = admin.MonitorConfigUpdate(grafana_dashboards=payload)
    put_resp = await admin.update_monitor_config(request=mock_request, body=body)
    assert put_resp.status_code == 200

    after = await admin.get_monitor_config(request=mock_request)
    assert _json_body(after)["grafana_dashboards_configured"] == ["overview"]


@pytest.mark.asyncio
async def test_enable_then_disable_dashboard_via_empty_string(mock_request, store):
    # Enable model_load with an explicit UID.
    enabled = {key: "" for key in ALL_DASHBOARD_KEYS}
    enabled["model_load"] = "ml-uid"
    await admin.update_monitor_config(
        request=mock_request, body=admin.MonitorConfigUpdate(grafana_dashboards=enabled)
    )
    assert "model_load" in store.get_configured_dashboard_keys()

    # Disable it again by submitting an explicit empty string, which must
    # overwrite the stale DB value so the tab falls back to "default".
    disabled = {key: "" for key in ALL_DASHBOARD_KEYS}
    await admin.update_monitor_config(
        request=mock_request,
        body=admin.MonitorConfigUpdate(grafana_dashboards=disabled),
    )
    configured = store.get_configured_dashboard_keys()
    assert "model_load" not in configured
    assert configured == ["overview"]
