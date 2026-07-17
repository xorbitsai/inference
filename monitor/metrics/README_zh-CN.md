# Xinference 监控指标

Xinference 集群的可观测性指标定义与采集说明。本文档涵盖由 Xinference 进程自身暴露的所有指标（不包含 DCGM Exporter 与 node_exporter 等外部组件的指标）。

## 监控体系概览

Xinference 同时提供两套并行且语义等价的指标导出体系，可按需选用或并行使用：

| 体系 | 实现库 | 导出方式 | 默认行为 | 开关 |
|------|--------|----------|----------|------|
| **Prometheus** | aioprometheus | Pull（HTTP `/metrics` 端点） | 默认开启 | 环境变量 `XINFERENCE_DISABLE_METRICS=1` 关闭 |
| **OpenTelemetry** | opentelemetry-python | Push（OTLP HTTP/gRPC） | 默认关闭 | 环境变量 `XINFERENCE_ENABLE_OTEL=true` 开启 |

两套体系采集相同的底层业务数据。Prometheus 适合传统 Pull 模式生态（Prometheus + Grafana），OpenTelemetry 适合 Push 模式的可观测性平台（Grafana Cloud、Datadog、阿里云 ARMS 等）。

## Prometheus 端点

Xinference 在以下两个进程分别暴露独立的 `/metrics` HTTP 端点：

| 进程 | 默认端口 | 指标作用域 |
|------|----------|------------|
| **Supervisor / xinference-local** | 与 API Server 同端口（默认 `9997`） | 集群全局视图 + Supervisor 自身指标 |
| **Worker** | 独立 metrics server（由 `--metrics-port` 指定） | 仅 Worker 自身指标 |

Prometheus 抓取配置示例：

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

> **指标隔离：** Supervisor 端点不会暴露 Worker 专属指标（如 token 计数），Worker 端点不会暴露 Supervisor 专属指标（如集群概览），以避免空 HELP/TYPE 头。

> **集群标签约定（`cluster`）：** Xinference 代码仅在 `xinference:build_info` 与 `xinference:config_info` 上写入 `cluster` 标签。若需对所有 `xinference:*` 指标及 HTTP 中间件计数器按集群过滤（多集群 Prometheus / 联邦部署），须在 Prometheus scrape 配置中用 `relabel_configs` 将 `cluster` 注入到全部 series。示例：
>
> ```yaml
> relabel_configs:
>   - target_label: cluster
>     replacement: <集群标识>
> ```

## OpenTelemetry 端点

通过环境变量配置 OTLP 导出（详见 `xinference/core/otel.py`）：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `XINFERENCE_ENABLE_OTEL` | 启用 OTEL | `false` |
| `XINFERENCE_OTLP_BASE_ENDPOINT` | OTLP 基础端点 | `http://localhost:4318` |
| `XINFERENCE_OTLP_TRACE_ENDPOINT` | Trace 端点（覆盖 base） | `<base>/v1/traces` |
| `XINFERENCE_OTLP_METRIC_ENDPOINT` | Metric 端点（覆盖 base） | `<base>/v1/metrics` |
| `XINFERENCE_OTLP_API_KEY` | Bearer 鉴权 Token | 空 |
| `XINFERENCE_OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` 或 `grpc` | `http/protobuf` |
| `XINFERENCE_OTEL_SAMPLING_RATE` | TraceIdRatio 采样率 | `0.1` |
| `XINFERENCE_OTEL_METRIC_EXPORT_INTERVAL` | 指标导出间隔（ms） | `60000` |

启用后，除业务 Trace 外，Worker 资源指标（CPU / 内存 / GPU）会以 ObservableGauge 形式由 Supervisor 周期性推送至 OTLP Metric 端点，指标命名规则为 `xinference.worker.*`。

---

## 指标清单

### 一、Worker 端指标（仅在 Worker `/metrics` 暴露）

Worker 端指标主要反映**单节点上的模型推理质量与服务负载**。

#### 1.1 LLM 推理指标

| 指标名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `xinference:generate_tokens_total` | Counter | tokens | LLM 累计生成的 token 总数 |
| `xinference:input_tokens_total_counter` | Counter | tokens | LLM 累计输入 token 总数 |
| `xinference:output_tokens_total_counter` | Counter | tokens | LLM 累计输出 token 总数 |
| `xinference:time_to_first_token_seconds` | Histogram | seconds | LLM 首 Token 延迟（TTFT）。分桶：0.05、0.1、0.25、0.5、1、2.5、5、10、30、+Inf |

**用途：**
- 生成吞吐（tokens/s）= `rate(generate_tokens_total)`
- 首 Token 延迟 P95 = `histogram_quantile(0.95, rate(time_to_first_token_seconds_bucket))`
- 评估模型推理性能与用户体感响应速度

#### 1.2 模型请求质量指标（适用于所有模型类型）

| 指标名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `xinference:model_request_total` | Counter | requests | 模型请求总数（按 `model_uid`、`replica_index`、`model_type`、`worker_address` 标签区分） |
| `xinference:model_request_errors_total` | Counter | requests | 模型请求失败总数 |
| `xinference:model_request_duration_seconds` | Histogram | seconds | 请求耗时分布。分桶：0.01、0.025、0.05、0.1、0.25、0.5、1、2.5、5、10、30、60、120、+Inf |
| `xinference:model_serve_count` | Gauge | requests | 当前正在处理的并发请求数 |
| `xinference:model_request_limit` | Gauge | requests | 模型配置的最大并发请求上限 |
| `xinference:model_last_load_duration_seconds` | Gauge | seconds | 最近一次模型加载耗时（含权重下载与初始化） |

**标签（与 Supervisor 端统一）：** `model_uid`（base uid，不含副本后缀）、`replica_index`（0 基，单副本=`0`）、`model_type`（`LLM`/`embedding`/`rerank`/`image`/`audio`/`video`）、`worker_address`、`model_name`、`engine`、`format`、`quantization`、`gpu_index`。

**用途：**
- QPS = `rate(model_request_total)`
- 错误率 = `rate(model_request_errors_total) / rate(model_request_total)`（SLO 核心）
- P50 / P95 / P99 延迟 = `histogram_quantile(...)`
- 当前并发负载率 = `model_serve_count / model_request_limit`（用于弹性扩缩容）
- 冷启动耗时 = `model_last_load_duration_seconds`

---

### 二、Supervisor 端指标（仅在 Supervisor `/metrics` 暴露）

Supervisor 端指标提供**集群全局视图**，包含节点资源、模型分布、生命周期与安全审计。

#### 2.1 集群与节点

| 指标名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `xinference:supervisor_uptime_seconds` | Gauge | seconds | Supervisor 进程运行时长 |
| `xinference:workers_total` | Gauge | count | 当前在线 Worker 数量 |
| `xinference:models_loaded_total` | Gauge | count | 已加载模型数量（按 `model_type` 标签聚合） |

**用途：**
- `workers_total` 突降触发 Worker 失联告警
- `models_loaded_total` 反映集群模型规模
- `supervisor_uptime_seconds` 用于识别 Supervisor 重启事件

#### 2.2 Worker 资源（每 worker_address 一个时间序列）

| 指标名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `xinference:worker_cpu_utilization` | Gauge | 0–1 | Worker CPU 使用率 |
| `xinference:worker_memory_used_bytes` | Gauge | bytes | Worker 已用内存 |
| `xinference:worker_memory_total_bytes` | Gauge | bytes | Worker 总内存 |
| `xinference:worker_gpu_utilization_percent` | Gauge | 0–100 | Worker GPU 利用率 |
| `xinference:worker_gpu_memory_used_bytes` | Gauge | bytes | Worker GPU 已用显存 |
| `xinference:worker_gpu_memory_total_bytes` | Gauge | bytes | Worker GPU 总显存 |

**标签：** `worker_address`，GPU 类指标附加 `gpu_index`、`gpu_name`。

**用途：**
- GPU 显存使用率 = `worker_gpu_memory_used_bytes / worker_gpu_memory_total_bytes`（高显存告警依据）
- 资源水位看板：CPU / 内存 / GPU 利用率实时趋势
- 容量规划：识别资源瓶颈 Worker

> **陈旧序列自动清理：** 当 Worker 下线或 GPU 被移除时，对应标签的时间序列会从 `/metrics` 中删除，避免 Prometheus 持续展示死样本。

#### 2.3 模型元信息（Info 类 Gauge，值恒为 1，信息承载于标签）

| 指标名 | 标签 | 说明 |
|--------|------|------|
| `xinference:model_info` | `model_uid`, `model_name`, `model_type`, `worker_address`, `replica_on_worker`, `replica_total` | 运行中模型的副本分布信息 |
| `xinference:model_status` | `model_uid`, `model_name`, `status` | 模型生命周期状态。`status` 取自 `LaunchStatus` 枚举：`CREATING` / `UPDATING` / `LOADING` / `READY` / `ERROR` / `TERMINATING` / `TERMINATED`。`/metrics` 活跃态主要为 `CREATING` / `LOADING` / `READY` / `ERROR`（`TERMINATED` 副本会被剔除；瞬态可见 `UPDATING` / `TERMINATING`） |
| `xinference:model_gpu_binding` | `model_uid`, `model_name`, `model_type`, `worker_address`, `gpu_index`, `replica_index` | 每个副本绑定的 GPU 列表 |
| `xinference:model_gpu_memory_used_bytes` | `model_uid`, `model_name`, `model_type`, `gpu_index`, `worker_address`, `replica_index` | 每个副本进程实际占用的 GPU 显存 |
| `xinference:model_unexpected_termination` | `model_uid`, `model_name`, `replica_index` | 因 Worker 故障异常下线的副本（redeploy 后自动清除） |

**用途：**
- `model_info` 与 `model_gpu_binding` 联合查询，回答"某模型某副本部署在哪块 GPU 上"
- `model_gpu_memory_used_bytes` 用于模型显存开销审计与模型规格对比
- `model_unexpected_termination == 1` 触发副本异常告警
- Info 类指标符合 Prometheus `info` metric 惯例，便于在 Grafana 中与其他指标做 label join

#### 2.4 API Key 安全审计

| 指标名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| `xinference:api_key_requests_total` | Counter | requests | API Key 鉴权接口请求总数。`status` 标签 ∈ {`success`, `model_not_found`, `error`, `denied`}（API Key 无权访问所请求模型时为 denied；其余情况下响应码 <400 为 success、404 为 model_not_found、其余为 error） |
| `xinference:api_key_request_duration_seconds` | Histogram | seconds | 鉴权接口耗时分布 |
| `xinference:api_keys_active_total` | Gauge | count | 当前有效的 API Key 数量 |
| `xinference:api_keys_expired_total` | Gauge | count | 已过期的 API Key 数量 |
| `xinference:banned_ips_total` | Gauge | count | 当前被封禁的 IP 数量 |
| `xinference:banned_keys_total` | Gauge | count | 当前被封禁的 (IP, Key) 对数量 |

**用途：**
- 鉴权接口 QPS 与延迟监控，评估鉴权瓶颈
- 封禁数量突增触发安全告警（疑似恶意攻击）
- 有效 Key 数量用于 API 资产盘点

#### 2.5 元信息（Build / Config Info）

| 指标名 | 标签 | 说明 |
|--------|------|------|
| `xinference:build_info` | `version`, `python_version`, `cluster`, `xinference_role`（`supervisor` / `worker`）, `worker_address`, `supervisor_address` | 版本与运行时信息 |
| `xinference:config_info` | `xinference_home`, `xinference_role`, `cluster`, `worker_address`, `supervisor_address` | 配置信息（如 XINFERENCE_HOME） |

**用途：**
- 多集群场景下区分版本、集群、角色
- 与业务指标做 join，支持按版本分组对比

#### 2.6 HTTP 中间件指标（仅 Supervisor 端）

由 `aioprometheus` 的 `MetricsMiddleware` 自动产生（在 Supervisor API Server 上挂载），用于 API 层 HTTP 健康 / 错误率监控。这些指标**不带 `xinference:` 前缀**，仅 Supervisor 端暴露（Worker 的 metrics server 是独立 FastAPI app，未挂该 middleware）。

| 指标名 | 类型 | 标签 | 说明 |
|--------|------|------|------|
| `requests_total_counter` | Counter | `method`, `path` | API 请求总数 |
| `responses_total_counter` | Counter | `method`, `path` | API 响应总数 |
| `exceptions_total_counter` | Counter | `method`, `path` | 抛出异常的请求数 |
| `status_codes_counter` | Counter | `method`, `path`, `status_code` | 按状态码统计的响应数（`status_code` 为完整码如 `200`/`404`/`500`，未按组归并） |

**用途：**
- API 错误率 = `sum without(status_code) (rate(status_codes_counter{status_code=~"4..|5.."}[5m])) / rate(requests_total_counter[5m])`
- API QPS = `sum(rate(requests_total_counter[5m]))`
- 异常请求数突增 = `rate(exceptions_total_counter[5m])`

> **`cluster` 标签：** 这些中间件指标由 Xinference 代码产生但本身不含 `cluster` 标签，需通过 Prometheus scrape relabel 注入（见上文「集群标签约定」）。

---

### 三、OpenTelemetry 指标（仅当 OTEL 启用时）

OTEL 指标与 Supervisor 端 Prometheus 指标语义等价，命名采用 OTLP 标准点分格式：

| OTEL 指标名 | 类型 | 单位 | 对应 Prometheus 指标 |
|-------------|------|------|----------------------|
| `xinference.worker.cpu.utilization` | ObservableGauge | 1 | `xinference:worker_cpu_utilization` |
| `xinference.worker.cpu.count` | ObservableGauge | cores | — |
| `xinference.worker.memory.used` | ObservableGauge | bytes | `xinference:worker_memory_used_bytes` |
| `xinference.worker.memory.total` | ObservableGauge | bytes | `xinference:worker_memory_total_bytes` |
| `xinference.worker.gpu.utilization` | ObservableGauge | % | `xinference:worker_gpu_utilization_percent` |
| `xinference.worker.gpu.memory.used` | ObservableGauge | bytes | `xinference:worker_gpu_memory_used_bytes` |
| `xinference.worker.gpu.memory.total` | ObservableGauge | bytes | `xinference:worker_gpu_memory_total_bytes` |
| `xinference.worker.gpu.memory.free` | ObservableGauge | bytes | — |

**标签：** `worker_address`，GPU 类指标附加 `gpu_index`、`gpu_name`。

**推送机制：** 由 Supervisor 端 `ClusterMetricsCollector` 在收到 Worker 周期性上报的 `node_info` 时刷新缓存，再由 OTEL SDK 的 ObservableGauge 回调主动拉取并推送至 OTLP Metric 端点。

---

## 指标用途矩阵

| 运营场景 | 关键指标 | 对应面板 |
|----------|----------|----------|
| **SLO 看板** | `time_to_first_token_seconds`、`model_request_duration_seconds`、`model_request_errors_total / model_request_total` | LLM SLO |
| **每模型 / 每副本负载** | `model_request_total` by (model_uid, replica_index, worker_address)、`model_serve_count`、`model_gpu_memory_used_bytes` | 模型负载 |
| **弹性扩缩容** | `model_serve_count / model_request_limit`、`models_loaded_total` | 模型负载 / 概览 |
| **故障检测与告警** | `workers_total`、`model_unexpected_termination`、`model_last_load_duration_seconds` | 概览 |
| **资源容量规划** | `worker_cpu_utilization`、`worker_memory_used_bytes`、`worker_gpu_utilization_percent`、`worker_gpu_memory_used_bytes` | 主机 / GPU |
| **安全审计** | `banned_ips_total`、`banned_keys_total`、`api_key_requests_total`、`api_keys_active_total` | 安全 |
| **资产盘点** | `model_info`、`model_gpu_binding`、`model_gpu_memory_used_bytes`、`build_info` | 模型负载 |

## 配套资源

- **Grafana 面板：** `monitor/dashboard/xinference-grafana-dashboard-*-{lang}.json`（6 个子面板 × 4 语言）
- **告警规则：** `monitor/alert/rules*.yml`
- **GPU 硬件指标：** `monitor/dashboard/dcgm-custom-metrics.csv`（依赖 DCGM Exporter）
- **日志采集：** `monitor/filebeat/`

> **单 Prometheus 实例前提：** 模型负载面板使用 Grafana Transform 在同一面板内 join Worker 侧和 Supervisor 侧指标。此方式要求两组指标由**同一 Prometheus 实例**抓取（使用不同 `job` 标签）。多 Prometheus 或联邦部署场景下，Worker 和 Supervisor 面板将独立展示，无法跨实例 join。

## 源码位置

- Prometheus 指标定义：`xinference/core/metrics.py`
- OpenTelemetry 初始化：`xinference/core/otel.py`
- REST API 中的 `/metrics` 路由注册：`xinference/api/restful_api.py`
- Worker 侧指标埋点：`xinference/core/model.py`
