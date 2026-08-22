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

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml  # type: ignore[import-untyped]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_signature(dashboard: Dict[str, Any]) -> Dict[int, List[Dict[str, str]]]:
    signature: Dict[int, List[Dict[str, str]]] = {}
    for panel in dashboard.get("panels", []):
        if panel.get("type") == "row":
            continue
        targets = []
        for target in panel.get("targets", []):
            targets.append(
                {
                    "refId": str(target.get("refId", "")),
                    "expr": str(target.get("expr", "")),
                }
            )
        signature[int(panel["id"])] = targets
    return signature


def _all_filebeat_inputs(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    inputs = config.get("filebeat.inputs")
    assert isinstance(inputs, list)
    return inputs


def test_token_router_dashboards_keep_localized_layout_and_promql_in_sync() -> None:
    paths = sorted(
        (_REPO_ROOT / "monitor/dashboard/token-router").glob(
            "xinference-grafana-dashboard-token-router-*.json"
        )
    )
    assert len(paths) == 4

    dashboards = [_load_json(path) for path in paths]
    assert {dashboard.get("uid") for dashboard in dashboards} == {
        "xinference-token-router"
    }

    expected = _panel_signature(dashboards[0])
    assert expected
    for dashboard in dashboards:
        assert _panel_signature(dashboard) == expected
        for targets in _panel_signature(dashboard).values():
            for target in targets:
                assert not target["refId"].startswith("$")
                assert target["refId"]


def test_localized_alert_rules_keep_semantics_in_sync() -> None:
    paths = [
        _REPO_ROOT / "monitor/alert/rules.yml",
        _REPO_ROOT / "monitor/alert/rules-zh-CN.yml",
        _REPO_ROOT / "monitor/alert/rules-ja.yml",
        _REPO_ROOT / "monitor/alert/rules-ko.yml",
    ]

    signatures = []
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        signature = []
        for group in config["groups"]:
            for rule in group.get("rules", []):
                signature.append(
                    {
                        "alert": rule["alert"],
                        "expr": rule["expr"],
                        "for": rule.get("for", ""),
                        "severity": rule.get("labels", {}).get("severity", ""),
                        "component": rule.get("labels", {}).get("component", ""),
                    }
                )
        signatures.append(signature)

    assert signatures[1:] == [signatures[0]] * 3
    by_name = {item["alert"]: item for item in signatures[0]}
    assert (
        "xinference:token_router_runtime_up == 1"
        in by_name["TokenRouterRuntimeConfigOutOfSync"]["expr"]
    )
    token_router_alerts = {
        name: item for name, item in by_name.items() if name.startswith("TokenRouter")
    }
    assert len(token_router_alerts) == 10
    assert {item["component"] for item in token_router_alerts.values()} == {
        "token-router"
    }


def test_filebeat_configs_collect_router_json_logs_without_rotated_archives() -> None:
    paths = sorted((_REPO_ROOT / "monitor/filebeat").glob("filebeat-*.yml"))
    assert len(paths) == 6

    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        by_log_type = {
            item.get("fields", {}).get("log_type"): item
            for item in _all_filebeat_inputs(config)
        }
        for log_type, expected_path in (
            ("router_agent", "/data/inference/logs/router-agent/router.log*"),
            ("router_runtime", "/data/inference/logs/router-runtime/*/router.log*"),
        ):
            item = by_log_type[log_type]
            assert expected_path in item["paths"]
            assert item.get("exclude_files") == [r"\.gz$"]
            assert item.get("json.keys_under_root") is True


def test_token_router_http_sd_example_uses_authenticated_15_second_refresh() -> None:
    path = _REPO_ROOT / "monitor/metrics/prometheus-token-router-http-sd.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    scrape = config["scrape_configs"][0]
    discovery = scrape["http_sd_configs"][0]

    assert scrape["job_name"] == "xinference-token-router-runtime"
    assert discovery["url"].endswith(
        "/v1/monitor/prometheus/http-sd/token-router-runtimes"
    )
    assert discovery["refresh_interval"] == "15s"
    assert discovery["authorization"] == {
        "type": "Bearer",
        "credentials": "REPLACE_WITH_MONITOR_TOKEN",
    }


def test_alertmanager_inhibition_preserves_router_unavailable() -> None:
    path = _REPO_ROOT / "monitor/alert/alertmanager-inhibition.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = config["inhibit_rules"]
    assert rules

    for rule in rules:
        targets = " ".join(rule.get("target_matchers", []))
        assert "TokenRouterUnavailable" not in targets

    agent_rule = next(
        rule
        for rule in rules
        if 'alertname="TokenRouterAgentOffline"' in rule.get("source_matchers", [])
    )
    assert agent_rule["equal"] == ["node_id"]
