# Xinference Grafana Dashboards

Pre-built Grafana dashboards for monitoring Xinference clusters, organized into 6 focused sub-dashboards, each targeting a specific audience and use case.

## Files

### Sub-Dashboards (6 dashboards × 4 languages = 24 files)

| Dashboard | UID | File Pattern | Audience |
|-----------|-----|--------------|----------|
| Cluster Overview | `xinference-overview` | `xinference-grafana-dashboard-overview-{lang}.json` | SRE / Platform |
| Model Load | `xinference-model-load` | `xinference-grafana-dashboard-model-load-{lang}.json` | ML Ops / Infra |
| LLM Inference SLO | `xinference-llm-slo` | `xinference-grafana-dashboard-llm-slo-{lang}.json` | ML Ops / Business |
| GPU Resources | `xinference-gpu-resources` | `xinference-grafana-dashboard-gpu-resources-{lang}.json` | SRE / Infra |
| Host Resources | `xinference-host-resources` | `xinference-grafana-dashboard-host-resources-{lang}.json` | SRE |
| Security Audit | `xinference-security-audit` | `xinference-grafana-dashboard-security-audit-{lang}.json` | SecOps |

Languages: `en`, `ja`, `ko`, `zh-CN`.

### Other Files

| File | Description |
|------|-------------|
| `dcgm-custom-metrics.csv` | Custom DCGM Exporter metrics configuration |
| `dcgm-exporter.yml` | Docker Compose service definition for DCGM Exporter |

## Quick Start

### 1. Import Dashboards into Grafana

**Manual import:**

1. Open Grafana → Dashboards → Import
2. Upload or paste the content of each dashboard file (e.g., `xinference-grafana-dashboard-overview-en.json`)
3. Select your Prometheus data source
4. Click Import
5. Repeat for all 6 sub-dashboards

**Automated provisioning:**

Create `provisioning/dashboards/xinference.yml`:

```yaml
apiVersion: 1

providers:
  - name: xinference
    orgId: 1
    folder: Xinference
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /path/to/monitor/dashboard
      foldersFromFilesStructure: false
```

Place all dashboard JSON files in the configured path. Grafana will auto-import and update them.

### 2. Deploy DCGM Exporter (GPU Monitoring)

On each GPU worker node:

```bash
docker compose -f dcgm-exporter.yml up -d
```

This deploys `nvidia/dcgm-exporter` with the custom metrics defined in `dcgm-custom-metrics.csv`, exposing metrics on port `9400`.

**Prerequisites:**
- NVIDIA GPU with driver installed
- Docker with NVIDIA Container Toolkit (`nvidia-docker2` or CDI)
- The `xinference` Docker network must exist: `docker network create xinference`

### 3. Configure Prometheus

Add scrape targets for:

- **Xinference Supervisor/Worker** (default port from Xinference metrics endpoint)
- **DCGM Exporter**: `<worker-host>:9400`
- **node_exporter**: `<host>:9100` (for host resource panels)

## Dashboard Details

### 1. Cluster Overview (`xinference-overview`)

Cluster health, Supervisor API quality, and alerting.

| Panel | Key Metrics |
|-------|-------------|
| Supervisor uptime | `supervisor_uptime_seconds` |
| Online Workers | `workers_total` |
| Models by type | `models_loaded_total` |
| Model unexpected termination | `model_unexpected_termination` |
| API QPS / errors | `supervisor_http_requests_total` |
| Active alerts | Alertmanager integration |

### 2. Model Load (`xinference-model-load`)

Per-model and per-replica view covering **all model types** (LLM, embedding, rerank, image, audio, video). Combines request load and GPU memory in one place.

| Panel | Key Metrics |
|-------|-------------|
| Per-type QPS / P95 / error rate | `model_request_total`, `model_request_duration_seconds` |
| Per-type total VRAM | `model_gpu_memory_used_bytes` |
| Per-model overview table | QPS, error rate, P95, concurrency, VRAM, GPU binding |
| Per-replica detail table | QPS, concurrency, VRAM per (model, worker) |
| Trend charts | QPS, concurrency ratio, VRAM, P95, error rate over time |

> **Label alignment:** Both Worker-side and Supervisor-side metrics now use unified labels (`model_uid`, `worker_address`, `model_type`, and `replica_index`). This allows direct PromQL joins (e.g., `* on (model_uid, replica_index, worker_address)`) within panels without requiring Grafana Transform hacks. This requires both Worker and Supervisor metrics to reside in the **same Prometheus instance** (different `job` labels).

### 3. LLM Inference SLO (`xinference-llm-slo`)

LLM-specific quality metrics (`model_type="LLM"` only). No per-model breakdown (moved to Model Load dashboard).

| Panel | Key Metrics |
|-------|-------------|
| TTFT P50/P95/P99 | `time_to_first_token_seconds_bucket` |
| Request duration P50/P95/P99 | `model_request_duration_seconds_bucket` |
| Token throughput | `generate_tokens_total`, `input_tokens_total_counter`, `output_tokens_total_counter` |
| Token amplification ratio | output / input tokens |

### 4. GPU Resources (`xinference-gpu-resources`)

Worker-level GPU utilization and DCGM hardware health. Model-level panels moved to Model Load dashboard.

| Panel | Key Metrics |
|-------|-------------|
| GPU utilization | `worker_gpu_utilization_percent` or `DCGM_FI_DEV_GPU_UTIL` |
| GPU memory | `worker_gpu_memory_used_bytes` |
| Active requests per Worker | `model_serve_count` |
| DCGM: temperature, power, PCIe errors, clock, health | DCGM Exporter metrics |

### 5. Host Resources (`xinference-host-resources`)

Supervisor and Worker host-level metrics. Shows both Xinference native gauges and node_exporter metrics side by side.

| Panel | Key Metrics |
|-------|-------------|
| CPU utilization | `worker_cpu_utilization` (Xinference) + `node_cpu_seconds_total` (node_exporter) |
| Memory | `worker_memory_used_bytes` / `worker_memory_total_bytes` (Xinference) + `node_memory_*` (node_exporter) |
| Disk, Swap, Network, I/O | node_exporter metrics |

### 6. Security Audit (`xinference-security-audit`)

API Key and ban statistics. **Only available when `XINFERENCE_AUTH_ADVANCED=true`.**

| Panel | Key Metrics |
|-------|-------------|
| Active / expired API Keys | `api_keys_active_total`, `api_keys_expired_total` |
| Banned IPs / (IP, Key) pairs | `banned_ips_total`, `banned_keys_total` |
| API Key request QPS / latency | `api_key_requests_total`, `api_key_request_duration_seconds` |
| Top users / top models | by API Key request count |

## DCGM Custom Metrics

The `dcgm-custom-metrics.csv` includes:

- **Base metrics**: GPU utilization, memory, temperature, power, PCIe errors, encoder/decoder usage
- **Enhanced metrics**: Power limits, fan speed, XID errors, performance state, clock frequencies, PCIe bandwidth, profiling (Volta+)

## Requirements

- Grafana >= 9.0
- Prometheus data source (single instance for both Worker and Supervisor metrics)
- Xinference with metrics enabled
- DCGM Exporter for GPU hardware panels
- node_exporter for host resource panels
