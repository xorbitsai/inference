# Xinference Monitoring Metrics

Observability metrics exposed by the Xinference cluster. This document covers all metrics produced by Xinference processes themselves; it does not include external exporters such as DCGM Exporter or node_exporter.

## Monitoring Systems Overview

Xinference provides two semantically equivalent metric export systems that can be used independently or in parallel:

| System | Library | Export Mode | Default | Toggle |
|--------|---------|-------------|---------|--------|
| **Prometheus** | aioprometheus | Pull (HTTP `/metrics` endpoint) | Enabled | Set `XINFERENCE_DISABLE_METRICS=1` to disable |
| **OpenTelemetry** | opentelemetry-python | Push (OTLP HTTP/gRPC) | Disabled | Set `XINFERENCE_ENABLE_OTEL=true` to enable |

Both systems collect the same underlying business data. Prometheus is suited for the traditional pull-based stack (Prometheus + Grafana), while OpenTelemetry fits push-based observability platforms (Grafana Cloud, Datadog, etc.).

## Prometheus Endpoint

Xinference exposes independent `/metrics` HTTP endpoints on two processes:

| Process | Default Port | Metric Scope |
|---------|--------------|--------------|
| **Supervisor / xinference-local** | Same as API Server (default `9997`) | Cluster-wide view + Supervisor-specific metrics |
| **Worker** | Dedicated metrics server (set via `--metrics-port`) | Worker-specific metrics only |

Prometheus scrape configuration example:

```yaml
scrape_configs:
  - job_name: xinference-supervisor
    static_configs:
      - targets: ['<supervisor-host>:9997']
        labels:
          role: supervisor

  - job_name: xinference-worker
    static_configs:
      - targets: ['<worker-host>:<metrics-port>']
        labels:
          role: worker
```

> **Metric isolation:** The Supervisor endpoint never exposes Worker-only metrics (e.g. token counters), and the Worker endpoint never exposes Supervisor-only metrics (e.g. cluster overview), avoiding empty HELP/TYPE headers.

> **Cluster label convention (`cluster`):** Xinference code writes the `cluster` label only on `xinference:build_info` and `xinference:config_info`. To filter all `xinference:*` metrics and HTTP middleware counters by cluster (multi-cluster Prometheus / federation), inject `cluster` onto every series via `relabel_configs` in the Prometheus scrape config. Example:
>
> ```yaml
> relabel_configs:
>   - target_label: cluster
>     replacement: <cluster-id>
> ```

## OpenTelemetry Endpoint

OTLP export is configured via environment variables (see `xinference/core/otel.py`):

| Variable | Description | Default |
|----------|-------------|---------|
| `XINFERENCE_ENABLE_OTEL` | Enable OTEL | `false` |
| `XINFERENCE_OTLP_BASE_ENDPOINT` | Base OTLP endpoint | `http://localhost:4318` |
| `XINFERENCE_OTLP_TRACE_ENDPOINT` | Override trace endpoint | `<base>/v1/traces` |
| `XINFERENCE_OTLP_METRIC_ENDPOINT` | Override metric endpoint | `<base>/v1/metrics` |
| `XINFERENCE_OTLP_API_KEY` | Bearer auth token | empty |
| `XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` or `grpc` | `http/protobuf` |
| `XINFERENCE_OTEL_SAMPLING_RATE` | TraceIdRatio sampling rate | `0.1` |
| `XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL` | Metric export interval (ms) | `60000` |

When enabled, in addition to business traces, Worker resource metrics (CPU / memory / GPU) are exported as ObservableGauges by the Supervisor via OTLP Metric endpoint, using the `xinference.worker.*` naming convention.

---

## Metric Inventory

### 1. Worker Metrics (exposed on Worker `/metrics` only)

Worker metrics focus on **per-node inference quality and service load**.

#### 1.1 LLM Inference Metrics

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `xinference:generate_tokens_total` | Counter | tokens | Total number of generated tokens |
| `xinference:input_tokens_total_counter` | Counter | tokens | Total number of input tokens |
| `xinference:output_tokens_total_counter` | Counter | tokens | Total number of output tokens |
| `xinference:time_to_first_token_seconds` | Histogram | seconds | Time to first token (TTFT). Buckets: 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, +Inf |

**Use cases:**
- Generation throughput (tokens/s) = `rate(generate_tokens_total)`
- TTFT P95 = `histogram_quantile(0.95, rate(time_to_first_token_seconds_bucket))`
- Assess inference performance and perceived user response speed

#### 1.2 Model Request Quality Metrics (all model types)

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `xinference:model_request_total` | Counter | requests | Total model requests (labeled by `model_uid`, `replica_index`, `model_type`, `worker_address`) |
| `xinference:model_request_errors_total` | Counter | requests | Total failed model requests |
| `xinference:model_request_duration_seconds` | Histogram | seconds | Request duration distribution. Buckets: 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, +Inf |
| `xinference:model_serve_count` | Gauge | requests | Number of requests currently being served |
| `xinference:model_request_limit` | Gauge | requests | Maximum concurrent request limit for the model |
| `xinference:model_last_load_duration_seconds` | Gauge | seconds | Duration of the last model load (including weight download and initialization) |

**Labels (unified with Supervisor side):** `model_uid` (base uid, no replica suffix), `replica_index` (0-based, `0` for single replica), `model_type` (`LLM`/`embedding`/`rerank`/`image`/`audio`/`video`), `worker_address`, `model_name`, `engine`, `format`, `quantization`, `gpu_index`.

**Use cases:**
- QPS = `rate(model_request_total)`
- Error rate = `rate(model_request_errors_total) / rate(model_request_total)` (core SLO)
- P50 / P95 / P99 latency = `histogram_quantile(...)`
- Concurrency load ratio = `model_serve_count / model_request_limit` (drives autoscaling)
- Cold-start latency = `model_last_load_duration_seconds`

---

### 2. Supervisor Metrics (exposed on Supervisor `/metrics` only)

Supervisor metrics provide a **cluster-wide view**, covering node resources, model distribution, lifecycle, and security auditing.

#### 2.1 Cluster and Nodes

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `xinference:supervisor_uptime_seconds` | Gauge | seconds | Supervisor process uptime |
| `xinference:workers_total` | Gauge | count | Number of online Workers |
| `xinference:models_loaded_total` | Gauge | count | Number of loaded models (labeled by `model_type`) |

**Use cases:**
- A sudden drop in `workers_total` triggers Worker-offline alerts
- `models_loaded_total` reflects cluster model scale
- `supervisor_uptime_seconds` helps detect Supervisor restart events

#### 2.2 Worker Resources (one time series per `worker_address`)

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `xinference:worker_cpu_utilization` | Gauge | 0–1 | Worker CPU utilization |
| `xinference:worker_memory_used_bytes` | Gauge | bytes | Worker memory used |
| `xinference:worker_memory_total_bytes` | Gauge | bytes | Worker total memory |
| `xinference:worker_gpu_utilization_percent` | Gauge | 0–100 | Worker GPU utilization |
| `xinference:worker_gpu_memory_used_bytes` | Gauge | bytes | Worker GPU memory used |
| `xinference:worker_gpu_memory_total_bytes` | Gauge | bytes | Worker GPU total memory |

**Labels:** `worker_address`; GPU metrics additionally carry `gpu_index` and `gpu_name`.

**Use cases:**
- GPU memory usage ratio = `worker_gpu_memory_used_bytes / worker_gpu_memory_total_bytes` (basis for high-memory alerts)
- Resource dashboard: real-time CPU / memory / GPU utilization trends
- Capacity planning: identify bottleneck Workers

> **Automatic stale-series cleanup:** When a Worker goes offline or a GPU is removed, the corresponding label set is removed from `/metrics`, preventing Prometheus from displaying dead samples.

#### 2.3 Model Metadata (Info-style Gauges, value always 1, info carried in labels)

| Metric | Labels | Description |
|--------|--------|-------------|
| `xinference:model_info` | `model_uid`, `model_name`, `model_type`, `worker_address`, `replica_on_worker`, `replica_total` | Running model replica distribution |
| `xinference:model_status` | `model_uid`, `model_name`, `status` | Model lifecycle status. `status` comes from the `LaunchStatus` enum: `CREATING` / `UPDATING` / `LOADING` / `READY` / `ERROR` / `TERMINATING` / `TERMINATED`. The live set on `/metrics` is mostly `CREATING` / `LOADING` / `READY` / `ERROR` (`TERMINATED` replicas are dropped; `UPDATING` / `TERMINATING` appear transiently) |
| `xinference:model_gpu_binding` | `model_uid`, `model_name`, `model_type`, `worker_address`, `gpu_index`, `replica_index` | GPU list bound to each replica |
| `xinference:model_gpu_memory_used_bytes` | `model_uid`, `model_name`, `model_type`, `gpu_index`, `worker_address`, `replica_index` | Actual GPU memory used per replica process |
| `xinference:model_unexpected_termination` | `model_uid`, `model_name`, `replica_index` | Replica terminated due to Worker failure (auto-cleared on redeploy) |

**Use cases:**
- Join `model_info` and `model_gpu_binding` to answer "which GPU is a given replica deployed on"
- `model_gpu_memory_used_bytes` audits model GPU memory footprint and compares model sizes
- `model_unexpected_termination == 1` triggers replica-failure alerts
- Info-style metrics follow the Prometheus `info` metric convention, enabling label joins in Grafana

#### 2.4 API Key Security Auditing

| Metric | Type | Unit | Description |
|--------|------|------|-------------|
| `xinference:api_key_requests_total` | Counter | requests | Total API Key auth requests. `status` label ∈ {`success`, `model_not_found`, `error`, `denied`} (`denied` means the API key cannot access the requested model; otherwise <400 is success, 404 is model_not_found, and other response codes are error) |
| `xinference:api_key_request_duration_seconds` | Histogram | seconds | Auth request duration distribution |
| `xinference:api_keys_active_total` | Gauge | count | Number of active API Keys |
| `xinference:api_keys_expired_total` | Gauge | count | Number of expired API Keys |
| `xinference:banned_ips_total` | Gauge | count | Number of currently banned IPs |
| `xinference:banned_keys_total` | Gauge | count | Number of currently banned (IP, Key) pairs |

**Use cases:**
- Auth endpoint QPS and latency to identify auth bottlenecks
- Spike in ban count triggers security alerts (suspected malicious activity)
- Active Key count for API asset inventory

#### 2.5 Metadata (Build / Config Info)

| Metric | Labels | Description |
|--------|--------|-------------|
| `xinference:build_info` | `version`, `python_version`, `cluster`, `xinference_role` (`supervisor` / `worker`), `worker_address`, `supervisor_address` | Version and runtime information |
| `xinference:config_info` | `xinference_home`, `xinference_role`, `cluster`, `worker_address`, `supervisor_address` | Configuration information (e.g. XINFERENCE_HOME) |

**Use cases:**
- Disambiguate version, cluster, and role in multi-cluster deployments
- Join with business metrics to compare across versions

#### 2.6 HTTP Middleware Metrics (Supervisor only)

Produced automatically by `aioprometheus`'s `MetricsMiddleware` (mounted on the Supervisor API Server) for HTTP health / error-rate monitoring at the API layer. These metrics **do not carry the `xinference:` prefix** and are exposed only on the Supervisor (the Worker metrics server is a standalone FastAPI app without this middleware).

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `requests_total_counter` | Counter | `method`, `path` | Total API requests |
| `responses_total_counter` | Counter | `method`, `path` | Total API responses |
| `exceptions_total_counter` | Counter | `method`, `path` | Requests that raised an exception |
| `status_codes_counter` | Counter | `method`, `path`, `status_code` | Responses by status code (`status_code` is the full code such as `200`/`404`/`500`; not grouped) |

**Use cases:**
- API error rate = `sum without(status_code) (rate(status_codes_counter{status_code=~"4..|5.."}[5m])) / rate(requests_total_counter[5m])`
- API QPS = `sum(rate(requests_total_counter[5m]))`
- Exception spike = `rate(exceptions_total_counter[5m])`

> **`cluster` label:** These middleware metrics are produced by Xinference but do not carry a `cluster` label; it must be injected via Prometheus scrape relabel (see "Cluster label convention" above).

---

### 3. OpenTelemetry Metrics (only when OTEL is enabled)

OTEL metrics are semantically equivalent to the Supervisor-side Prometheus metrics, using OTLP-standard dot-separated naming:

| OTEL Metric | Type | Unit | Prometheus Equivalent |
|-------------|------|------|-----------------------|
| `xinference.worker.cpu.utilization` | ObservableGauge | 1 | `xinference:worker_cpu_utilization` |
| `xinference.worker.cpu.count` | ObservableGauge | cores | — |
| `xinference.worker.memory.used` | ObservableGauge | bytes | `xinference:worker_memory_used_bytes` |
| `xinference.worker.memory.total` | ObservableGauge | bytes | `xinference:worker_memory_total_bytes` |
| `xinference.worker.gpu.utilization` | ObservableGauge | % | `xinference:worker_gpu_utilization_percent` |
| `xinference.worker.gpu.memory.used` | ObservableGauge | bytes | `xinference:worker_gpu_memory_used_bytes` |
| `xinference.worker.gpu.memory.total` | ObservableGauge | bytes | `xinference:worker_gpu_memory_total_bytes` |
| `xinference.worker.gpu.memory.free` | ObservableGauge | bytes | — |

**Labels:** `worker_address`; GPU metrics additionally carry `gpu_index` and `gpu_name`.

**Push mechanism:** The Supervisor-side `ClusterMetricsCollector` refreshes its cache upon receiving periodic `node_info` reports from Workers; the OTEL SDK's ObservableGauge callbacks then pull from the cache and push to the OTLP Metric endpoint.

---

## Use-Case Matrix

| Operations Scenario | Key Metrics | Dashboard |
|---------------------|-------------|-----------|
| **SLO Dashboard** | `time_to_first_token_seconds`, `model_request_duration_seconds`, `model_request_errors_total / model_request_total` | LLM SLO |
| **Per-Model / Per-Replica Load** | `model_request_total` by (model_uid, replica_index, worker_address), `model_serve_count`, `model_gpu_memory_used_bytes` | Model Load |
| **Autoscaling** | `model_serve_count / model_request_limit`, `models_loaded_total` | Model Load / Overview |
| **Failure Detection & Alerts** | `workers_total`, `model_unexpected_termination`, `model_last_load_duration_seconds` | Overview |
| **Resource Capacity Planning** | `worker_cpu_utilization`, `worker_memory_used_bytes`, `worker_gpu_utilization_percent`, `worker_gpu_memory_used_bytes` | Host / GPU |
| **Security Auditing** | `banned_ips_total`, `banned_keys_total`, `api_key_requests_total`, `api_keys_active_total` | Security |
| **Asset Inventory** | `model_info`, `model_gpu_binding`, `model_gpu_memory_used_bytes`, `build_info` | Model Load |

## Companion Resources

- **Grafana Dashboards:** `monitor/dashboard/xinference-grafana-dashboard-*-{lang}.json` (6 sub-dashboards × 4 languages)
- **Alert Rules:** `monitor/alert/rules*.yml`
- **GPU Hardware Metrics:** `monitor/dashboard/dcgm-custom-metrics.csv` (requires DCGM Exporter)
- **Log Collection:** `monitor/filebeat/`

> **Single Prometheus instance prerequisite:** The Model Load dashboard uses Grafana Transform to join Worker-side and Supervisor-side metrics within the same panel. This requires both sets of metrics to be scraped by the **same Prometheus instance** (with different `job` labels). Multi-Prometheus or federated deployments will see Worker and Supervisor panels independently without cross-join.

## Source Code Locations

- Prometheus metric definitions: `xinference/core/metrics.py`
- OpenTelemetry initialization: `xinference/core/otel.py`
- `/metrics` route registration in REST API: `xinference/api/restful_api.py`
- Worker-side metric instrumentation: `xinference/core/model.py`
