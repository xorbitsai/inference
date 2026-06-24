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

"""Unit tests for MonitorConfigStore."""

import threading

import pytest

from xinference.core.monitor_config_store import MonitorConfigStore


@pytest.fixture
def store(tmp_path):
    return MonitorConfigStore(str(tmp_path / "monitor_config.db"))


def test_init_creates_default_rows(store):
    all_cfg = store.get_all()
    assert "grafana_url" in all_cfg
    assert "grafana_datasource" in all_cfg
    assert "dashboard_overview" in all_cfg


def test_get_all_returns_defaults_when_empty(store):
    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == ""
    assert all_cfg["dashboard_overview"] == "xinference-overview"
    assert all_cfg["dashboard_model_load"] == "xinference-model-load"


def test_get_all_env_fallback(store, monkeypatch):
    monkeypatch.setenv("XINFERENCE_GRAFANA_URL", "http://grafana.test:3000")
    monkeypatch.setenv("XINFERENCE_GRAFANA_DATASOURCE", "MyProm")
    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == "http://grafana.test:3000"
    assert all_cfg["grafana_datasource"] == "MyProm"


def test_get_sources_default(store):
    sources = store.get_sources()
    assert sources["grafana_url"] == "default"
    assert sources["dashboard_overview"] == "default"


def test_get_sources_env(store, monkeypatch):
    monkeypatch.setenv("XINFERENCE_GRAFANA_URL", "http://env.url")
    sources = store.get_sources()
    assert sources["grafana_url"] == "env"


def test_update_and_get(store):
    store.update(
        {"grafana_url": "http://db.url", "cluster_name": "prod"}, username="admin"
    )
    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == "http://db.url"
    assert all_cfg["cluster_name"] == "prod"


def test_get_sources_db(store):
    store.update({"grafana_url": "http://db.url"}, username="admin")
    sources = store.get_sources()
    assert sources["grafana_url"] == "db"


def test_db_takes_priority_over_env(store, monkeypatch):
    monkeypatch.setenv("XINFERENCE_GRAFANA_URL", "http://env.url")
    store.update({"grafana_url": "http://db.url"}, username="admin")
    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == "http://db.url"


def test_get_dashboards(store):
    store.update(
        {"dashboard_overview": "custom-overview", "dashboard_model_load": "custom-ml"},
        username="admin",
    )
    dashboards = store.get_dashboards()
    assert dashboards["overview"] == "custom-overview"
    assert dashboards["model_load"] == "custom-ml"
    assert dashboards["llm_slo"] == "xinference-llm-slo"


def test_alert_datasource_fallback(store):
    store.update({"grafana_datasource": "MainProm"}, username="admin")
    all_cfg = store.get_all()
    assert all_cfg["grafana_alert_datasource"] == "MainProm"


def test_alert_datasource_explicit(store):
    store.update(
        {"grafana_datasource": "MainProm", "grafana_alert_datasource": "AlertProm"},
        username="admin",
    )
    all_cfg = store.get_all()
    assert all_cfg["grafana_alert_datasource"] == "AlertProm"


def test_reset_clears_all(store):
    store.update({"grafana_url": "http://db.url"}, username="admin")
    store.reset()
    sources = store.get_sources()
    assert sources["grafana_url"] == "default"
    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == ""


def test_reset_then_update(store):
    store.update({"grafana_url": "first"}, username="admin")
    store.reset()
    store.update({"grafana_url": "second"}, username="admin")
    all_cfg = store.get_all()
    assert all_cfg["grafana_url"] == "second"


def test_concurrent_writes(store):
    errors = []

    def writer(idx):
        try:
            store.update({f"cluster_name": f"cluster-{idx}"}, username=f"user-{idx}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    all_cfg = store.get_all()
    assert all_cfg["cluster_name"].startswith("cluster-")
