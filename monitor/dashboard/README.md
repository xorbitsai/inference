# Xinference Grafana Dashboard

Pre-built Grafana dashboard for monitoring Xinference clusters, including Supervisor API, Worker resources, GPU hardware (via DCGM), model service quality, and host-level metrics (via node_exporter).

## Files

| File | Description |
|------|-------------|
| `xinference-grafana-dashboard-en.json` | English localized dashboard |
| `xinference-grafana-dashboard-ja.json` | Japanese localized dashboard |
| `xinference-grafana-dashboard-ko.json` | Korean localized dashboard |
| `xinference-grafana-dashboard-zh-CN.json` | Refined Chinese localized dashboard |
| `dcgm-custom-metrics.csv` | Custom DCGM Exporter metrics configuration |
| `dcgm-exporter.yml` | Docker Compose service definition for DCGM Exporter |

## Quick Start

### 1. Import Dashboard into Grafana

1. Open Grafana → Dashboards → Import
2. Upload or paste the content of the desired locale file (e.g., `xinference-grafana-dashboard-en.json`)
3. Select your Prometheus data source
4. Click Import

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

## DCGM Custom Metrics

The `dcgm-custom-metrics.csv` includes:

- **Base metrics**: GPU utilization, memory, temperature, power, PCIe errors, encoder/decoder usage
- **Enhanced metrics**: Power limits, fan speed, XID errors, performance state, clock frequencies, PCIe bandwidth, profiling (Volta+)

## Dashboard Sections

| Section | Description |
|---------|-------------|
| Cluster Overview | Supervisor uptime, online workers, build/config info, models by type |
| Supervisor API Monitoring | API QPS, exceptions, HTTP error rate, status code distribution, top paths |
| Worker Resources | GPU utilization, GPU memory, CPU, RAM |
| GPU Hardware (DCGM) | Temperature, power draw/limit, PCIe errors, clock frequency, health status |
| Model Status & GPU Binding | Model deployment table, GPU binding relationships |
| Model Service Quality | QPS, error rate, P95 latency, concurrent requests |
| LLM Specific | Generation throughput (tokens/s), time to first token |
| Supervisor Host Resources | CPU, memory, disk, network, swap, I/O (via node_exporter) |
| Worker Host Resources | CPU, memory, disk, network, swap, I/O (via node_exporter) |

## Requirements

- Grafana >= 9.0
- Prometheus data source
- Xinference with metrics enabled
- DCGM Exporter for GPU hardware panels
- node_exporter for host resource panels
