#!/usr/bin/env python3
"""Generate 6 sub-dashboards × 4 languages = 24 Grafana JSON files.

Reads the original monolithic dashboard and redistributes panels into
6 focused dashboards with proper translations.
"""

import json
import os
from copy import deepcopy
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_FILE = os.path.join(SCRIPT_DIR, "xinference-grafana-dashboard-zh-CN.json")

LANGUAGES = ["zh-CN", "en", "ja", "ko"]

DASHBOARD_CONFIGS = {
    "overview": {
        "uid": "xinference-overview",
        "rows": ["集群概览", "Supervisor API 监控"],
        "new_panels": ["model_unexpected_termination"],
    },
    "model-load": {
        "uid": "xinference-model-load",
        "rows": ["模型状态与 GPU 绑定"],
        "new_panels": [
            "per_type_qps",
            "per_type_p95",
            "per_type_error_rate",
            "per_type_vram",
            "per_model_overview_table",
            "per_replica_table",
            "per_model_qps_trend",
            "per_model_concurrency_trend",
            "per_replica_vram_trend",
            "per_model_p95_trend",
            "per_model_error_trend",
        ],
    },
    "llm-slo": {
        "uid": "xinference-llm-slo",
        "rows": ["LLM 专属"],
        "new_panels": [
            "ttft_p50_p95_p99",
            "request_duration_p50_p95_p99",
        ],
    },
    "gpu-resources": {
        "uid": "xinference-gpu-resources",
        "rows": ["Worker GPU 资源与负载", "GPU 硬件监控（DCGM）"],
        "new_panels": [],
    },
    "host-resources": {
        "uid": "xinference-host-resources",
        "rows": [
            "Supervisor 主机资源（node_exporter）",
            "Worker 主机资源（node_exporter）",
        ],
        "new_panels": [
            "worker_cpu_utilization",
            "worker_memory_usage",
        ],
    },
    "security-audit": {
        "uid": "xinference-security-audit",
        "rows": ["安全审计"],
        "new_panels": [],
    },
}

TRANSLATIONS = {
    "zh-CN": {
        "overview": "集群概览",
        "model-load": "模型负载",
        "llm-slo": "LLM 推理 SLO",
        "gpu-resources": "GPU 资源与硬件",
        "host-resources": "主机资源",
        "security-audit": "安全审计",
        "model_unexpected_termination": "模型异常终止",
        "per_type_qps": "每类型 QPS",
        "per_type_p95": "每类型 P95 延迟",
        "per_type_error_rate": "每类型错误率",
        "per_type_vram": "每类型总显存",
        "per_model_overview_table": "每模型总览",
        "per_replica_table": "每副本明细",
        "per_model_qps_trend": "每模型 QPS 趋势",
        "per_model_concurrency_trend": "每模型并发率趋势",
        "per_replica_vram_trend": "每副本显存占用趋势",
        "per_model_p95_trend": "每模型 P95 延迟趋势",
        "per_model_error_trend": "每模型错误率趋势",
        "ttft_p50_p95_p99": "TTFT P50/P95/P99",
        "request_duration_p50_p95_p99": "请求延迟 P50/P95/P99",
        "worker_cpu_utilization": "Worker CPU 利用率 (Xinference)",
        "worker_memory_usage": "Worker 内存使用 (Xinference)",
    },
    "en": {
        "overview": "Cluster Overview",
        "model-load": "Model Load",
        "llm-slo": "LLM Inference SLO",
        "gpu-resources": "GPU Resources & Hardware",
        "host-resources": "Host Resources",
        "security-audit": "Security Audit",
        "集群概览": "Cluster Overview",
        "Supervisor API 监控": "Supervisor API Monitoring",
        "Worker GPU 资源与负载": "Worker GPU Resources & Load",
        "GPU 硬件监控（DCGM）": "GPU Hardware Monitoring (DCGM)",
        "模型状态与 GPU 绑定": "Model Status & GPU Binding",
        "LLM 专属": "LLM Specific",
        "非 LLM 模型专项": "Non-LLM Models",
        "Supervisor 主机资源（node_exporter）": "Supervisor Host Resources (node_exporter)",
        "Worker 主机资源（node_exporter）": "Worker Host Resources (node_exporter)",
        "安全审计": "Security Audit",
        "model_unexpected_termination": "Model Unexpected Termination",
        "per_type_qps": "Per-Type QPS",
        "per_type_p95": "Per-Type P95 Latency",
        "per_type_error_rate": "Per-Type Error Rate",
        "per_type_vram": "Per-Type Total VRAM",
        "per_model_overview_table": "Per-Model Overview",
        "per_replica_table": "Per-Replica Details",
        "per_model_qps_trend": "Per-Model QPS Trend",
        "per_model_concurrency_trend": "Per-Model Concurrency Trend",
        "per_replica_vram_trend": "Per-Replica VRAM Trend",
        "per_model_p95_trend": "Per-Model P95 Latency Trend",
        "per_model_error_trend": "Per-Model Error Rate Trend",
        "ttft_p50_p95_p99": "TTFT P50/P95/P99",
        "request_duration_p50_p95_p99": "Request Duration P50/P95/P99",
        "worker_cpu_utilization": "Worker CPU Utilization (Xinference)",
        "worker_memory_usage": "Worker Memory Usage (Xinference)",
    },
    "ja": {
        "overview": "クラスター概要",
        "model-load": "モデル負荷",
        "llm-slo": "LLM 推論 SLO",
        "gpu-resources": "GPU リソースとハードウェア",
        "host-resources": "ホストリソース",
        "security-audit": "セキュリティ監査",
        "集群概览": "クラスター概要",
        "Supervisor API 监控": "Supervisor API 監視",
        "Worker GPU 资源与负载": "Worker GPU リソースと負荷",
        "GPU 硬件监控（DCGM）": "GPU ハードウェア監視 (DCGM)",
        "模型状态与 GPU 绑定": "モデル状態と GPU バインディング",
        "LLM 专属": "LLM 専用",
        "非 LLM 模型专项": "非 LLM モデル",
        "Supervisor 主机资源（node_exporter）": "Supervisor ホストリソース (node_exporter)",
        "Worker 主机资源（node_exporter）": "Worker ホストリソース (node_exporter)",
        "安全审计": "セキュリティ監査",
        "model_unexpected_termination": "モデル予期しない終了",
        "per_type_qps": "タイプ別 QPS",
        "per_type_p95": "タイプ別 P95 レイテンシ",
        "per_type_error_rate": "タイプ別エラー率",
        "per_type_vram": "タイプ別合計 VRAM",
        "per_model_overview_table": "モデル別概要",
        "per_replica_table": "レプリカ別詳細",
        "per_model_qps_trend": "モデル別 QPS トレンド",
        "per_model_concurrency_trend": "モデル別同時実行トレンド",
        "per_replica_vram_trend": "レプリカ別 VRAM トレンド",
        "per_model_p95_trend": "モデル別 P95 レイテンシトレンド",
        "per_model_error_trend": "モデル別エラー率トレンド",
        "ttft_p50_p95_p99": "TTFT P50/P95/P99",
        "request_duration_p50_p95_p99": "リクエスト遅延 P50/P95/P99",
        "worker_cpu_utilization": "Worker CPU 使用率 (Xinference)",
        "worker_memory_usage": "Worker メモリ使用量 (Xinference)",
    },
    "ko": {
        "overview": "클러스터 개요",
        "model-load": "모델 부하",
        "llm-slo": "LLM 추론 SLO",
        "gpu-resources": "GPU 리소스 및 하드웨어",
        "host-resources": "호스트 리소스",
        "security-audit": "보안 감사",
        "集群概览": "클러스터 개요",
        "Supervisor API 监控": "Supervisor API 모니터링",
        "Worker GPU 资源与负载": "Worker GPU 리소스 및 부하",
        "GPU 硬件监控（DCGM）": "GPU 하드웨어 모니터링 (DCGM)",
        "模型状态与 GPU 绑定": "모델 상태 및 GPU 바인딩",
        "LLM 专属": "LLM 전용",
        "非 LLM 模型专项": "비 LLM 모델",
        "Supervisor 主机资源（node_exporter）": "Supervisor 호스트 리소스 (node_exporter)",
        "Worker 主机资源（node_exporter）": "Worker 호스트 리소스 (node_exporter)",
        "安全审计": "보안 감사",
        "model_unexpected_termination": "모델 예기치 않은 종료",
        "per_type_qps": "유형별 QPS",
        "per_type_p95": "유형별 P95 지연시간",
        "per_type_error_rate": "유형별 오류율",
        "per_type_vram": "유형별 총 VRAM",
        "per_model_overview_table": "모델별 개요",
        "per_replica_table": "레플리카별 상세",
        "per_model_qps_trend": "모델별 QPS 트렌드",
        "per_model_concurrency_trend": "모델별 동시성 트렌드",
        "per_replica_vram_trend": "레플리카별 VRAM 트렌드",
        "per_model_p95_trend": "모델별 P95 지연시간 트렌드",
        "per_model_error_trend": "모델별 오류율 트렌드",
        "ttft_p50_p95_p99": "TTFT P50/P95/P99",
        "request_duration_p50_p95_p99": "요청 지연시간 P50/P95/P99",
        "worker_cpu_utilization": "Worker CPU 사용률 (Xinference)",
        "worker_memory_usage": "Worker 메모리 사용량 (Xinference)",
    },
}


def load_original() -> Dict[str, Any]:
    with open(ORIGINAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_panels_by_row(panels: List[Dict]) -> Dict[str, List[Dict]]:
    """Group panels by their parent row title."""
    result: Dict[str, List[Dict]] = {}
    current_row = None
    for panel in panels:
        if panel.get("type") == "row":
            current_row = panel.get("title", "Unknown")
            result[current_row] = [panel]
        elif current_row:
            result[current_row].append(panel)
    return result


def create_dashboard_template(
    original: Dict[str, Any], uid: str, title: str, lang: str
) -> Dict[str, Any]:
    """Create a dashboard skeleton based on the original."""
    dashboard = deepcopy(original)
    dashboard["uid"] = uid
    dashboard["title"] = title
    dashboard["panels"] = []
    dashboard["version"] = 1
    dashboard["id"] = None

    # Add dashboard links to other sub-dashboards
    links = []
    for dash_key, cfg in DASHBOARD_CONFIGS.items():
        if cfg["uid"] != uid:
            links.append(
                {
                    "asDropdown": False,
                    "icon": "external link",
                    "includeVars": True,
                    "keepTime": True,
                    "tags": [],
                    "targetBlank": False,
                    "title": TRANSLATIONS[lang].get(dash_key, dash_key),
                    "tooltip": "",
                    "type": "link",
                    "url": f"/d/{cfg['uid']}",
                }
            )
    dashboard["links"] = links
    return dashboard


def create_new_panel(panel_id: int, panel_type: str, key: str, lang: str) -> Dict:
    """Create a new panel for missing metrics."""
    title = TRANSLATIONS[lang].get(key, key)

    if key == "model_unexpected_termination":
        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": "xinference:model_unexpected_termination",
                    "legendFormat": "{{model_name}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        }
    elif key == "per_type_qps":
        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": 'sum by (type) (rate(model_request_total{type=~"$model_type"}[5m]))',
                    "legendFormat": "{{type}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        }
    elif key == "per_type_p95":
        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": 'histogram_quantile(0.95, sum by (le, type) (rate(model_request_duration_seconds_bucket{type=~"$model_type"}[5m])))',
                    "legendFormat": "{{type}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
            "fieldConfig": {
                "defaults": {"unit": "s"},
            },
        }
    elif key == "per_type_error_rate":
        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": 'sum by (type) (rate(model_request_errors_total{type=~"$model_type"}[5m])) / sum by (type) (rate(model_request_total{type=~"$model_type"}[5m]))',
                    "legendFormat": "{{type}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
            "fieldConfig": {
                "defaults": {"unit": "percentunit"},
            },
        }
    elif key == "per_type_vram":
        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": 'sum by (model_type) (model_gpu_memory_used_bytes{model_type=~"$model_type"})',
                    "legendFormat": "{{model_type}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
            "fieldConfig": {
                "defaults": {"unit": "bytes"},
            },
        }
    elif key == "ttft_p50_p95_p99":
        return {
            "id": panel_id,
            "title": title,
            "type": "timeseries",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": 'histogram_quantile(0.50, sum by (le) (rate(time_to_first_token_seconds_bucket{type="llm"}[5m])))',
                    "legendFormat": "P50",
                    "refId": "A",
                },
                {
                    "expr": 'histogram_quantile(0.95, sum by (le) (rate(time_to_first_token_seconds_bucket{type="llm"}[5m])))',
                    "legendFormat": "P95",
                    "refId": "B",
                },
                {
                    "expr": 'histogram_quantile(0.99, sum by (le) (rate(time_to_first_token_seconds_bucket{type="llm"}[5m])))',
                    "legendFormat": "P99",
                    "refId": "C",
                },
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "fieldConfig": {
                "defaults": {"unit": "s"},
            },
        }
    elif key == "request_duration_p50_p95_p99":
        return {
            "id": panel_id,
            "title": title,
            "type": "timeseries",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": 'histogram_quantile(0.50, sum by (le) (rate(model_request_duration_seconds_bucket{type="llm"}[5m])))',
                    "legendFormat": "P50",
                    "refId": "A",
                },
                {
                    "expr": 'histogram_quantile(0.95, sum by (le) (rate(model_request_duration_seconds_bucket{type="llm"}[5m])))',
                    "legendFormat": "P95",
                    "refId": "B",
                },
                {
                    "expr": 'histogram_quantile(0.99, sum by (le) (rate(model_request_duration_seconds_bucket{type="llm"}[5m])))',
                    "legendFormat": "P99",
                    "refId": "C",
                },
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "fieldConfig": {
                "defaults": {"unit": "s"},
            },
        }
    elif key == "worker_cpu_utilization":
        return {
            "id": panel_id,
            "title": title,
            "type": "timeseries",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": "xinference:worker_cpu_utilization",
                    "legendFormat": "{{instance}}",
                    "refId": "A",
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "fieldConfig": {
                "defaults": {"unit": "percent"},
            },
        }
    elif key == "worker_memory_usage":
        return {
            "id": panel_id,
            "title": title,
            "type": "timeseries",
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [
                {
                    "expr": "xinference:worker_memory_used_bytes",
                    "legendFormat": "{{instance}} used",
                    "refId": "A",
                },
                {
                    "expr": "xinference:worker_memory_total_bytes",
                    "legendFormat": "{{instance}} total",
                    "refId": "B",
                },
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "fieldConfig": {
                "defaults": {"unit": "bytes"},
            },
        }
    else:
        # Placeholder for other new panels
        return {
            "id": panel_id,
            "title": title,
            "type": panel_type,
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "targets": [],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "description": f"TODO: Add PromQL for {key}",
        }


def translate_panel(panel: Dict, lang: str) -> Dict:
    """Translate panel title if needed."""
    if lang == "zh-CN":
        return panel
    panel = deepcopy(panel)
    title = panel.get("title", "")
    if title in TRANSLATIONS[lang]:
        panel["title"] = TRANSLATIONS[lang][title]
    return panel


def generate_dashboard(
    dash_key: str,
    lang: str,
    original: Dict[str, Any],
    panels_by_row: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """Generate a single dashboard JSON."""
    cfg = DASHBOARD_CONFIGS[dash_key]
    uid = cfg["uid"]
    title = TRANSLATIONS[lang].get(dash_key, dash_key)

    dashboard = create_dashboard_template(original, uid, title, lang)

    # Collect panels from specified rows
    panels: List[Dict] = []
    panel_id_counter = 400

    for row_title in cfg["rows"]:
        row_panels = panels_by_row.get(row_title, [])
        for p in row_panels:
            translated = translate_panel(p, lang)
            panels.append(translated)

    # Add new panels
    for new_key in cfg.get("new_panels", []):
        panel_id_counter += 1
        new_panel = create_new_panel(panel_id_counter, "stat", new_key, lang)
        panels.append(new_panel)

    dashboard["panels"] = panels
    return dashboard


def main():
    print(f"Loading original dashboard from {ORIGINAL_FILE}")
    original = load_original()
    panels_by_row = extract_panels_by_row(original["panels"])

    print(f"Found {len(panels_by_row)} rows with panels")

    generated_files = []

    for dash_key, cfg in DASHBOARD_CONFIGS.items():
        out_dir = os.path.join(SCRIPT_DIR, dash_key)
        os.makedirs(out_dir, exist_ok=True)
        for lang in LANGUAGES:
            dashboard = generate_dashboard(dash_key, lang, original, panels_by_row)
            filename = f"xinference-grafana-dashboard-{dash_key}-{lang}.json"
            filepath = os.path.join(out_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dashboard, f, ensure_ascii=False, indent=2)

            generated_files.append(filename)
            print(f"  Generated: {filename}")

    print(f"\nGenerated {len(generated_files)} files")

    # Verify UIDs are consistent across languages
    print("\nVerifying UID consistency...")
    for dash_key, cfg in DASHBOARD_CONFIGS.items():
        uids = set()
        for lang in LANGUAGES:
            filename = f"xinference-grafana-dashboard-{dash_key}-{lang}.json"
            filepath = os.path.join(SCRIPT_DIR, dash_key, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                uids.add(data["uid"])
        assert len(uids) == 1, f"UID mismatch for {dash_key}: {uids}"
        print(f"  {dash_key}: uid={uids.pop()} ✓")

    print("\nDone!")


if __name__ == "__main__":
    main()
