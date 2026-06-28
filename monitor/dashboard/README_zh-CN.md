# Xinference Grafana 监控面板

预构建的 Grafana 监控面板，用于监控 Xinference 集群，按受众和使用场景组织为 6 个聚焦子面板。

## 文件说明

### 子面板（6 面板 × 4 语言 = 24 文件）

| 面板 | UID | 文件名模式 | 主要受众 |
|------|-----|------------|----------|
| 集群概览 | `xinference-overview` | `xinference-grafana-dashboard-overview-{lang}.json` | SRE / 平台 |
| 模型负载 | `xinference-model-load` | `xinference-grafana-dashboard-model-load-{lang}.json` | ML Ops / 算力运维 |
| LLM 推理 SLO | `xinference-llm-slo` | `xinference-grafana-dashboard-llm-slo-{lang}.json` | ML Ops / 业务 |
| GPU 资源 | `xinference-gpu-resources` | `xinference-grafana-dashboard-gpu-resources-{lang}.json` | SRE / 算力运维 |
| 主机资源 | `xinference-host-resources` | `xinference-grafana-dashboard-host-resources-{lang}.json` | SRE |
| 安全审计 | `xinference-security-audit` | `xinference-grafana-dashboard-security-audit-{lang}.json` | SecOps |

语言：`en`、`ja`、`ko`、`zh-CN`。

### 其他文件

| 文件 | 说明 |
|------|------|
| `dcgm-custom-metrics.csv` | 自定义 DCGM Exporter 指标配置 |
| `dcgm-exporter.yml` | DCGM Exporter 的 Docker Compose 服务定义 |

## 快速开始

### 1. 导入面板到 Grafana

**手动导入：**

1. 打开 Grafana → Dashboards → Import
2. 上传或粘贴面板文件内容（如 `xinference-grafana-dashboard-overview-zh-CN.json`）
3. 选择你的 Prometheus 数据源
4. 点击 Import
5. 对 6 个子面板重复以上步骤

**自动 Provisioning：**

创建 `provisioning/dashboards/xinference.yml`：

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

将所有面板 JSON 文件放在配置的路径下，Grafana 会自动导入并更新。

### 2. 部署 DCGM Exporter（GPU 监控）

在每个 GPU Worker 节点上执行：

```bash
docker compose -f dcgm-exporter.yml up -d
```

这将部署 `nvidia/dcgm-exporter`，使用 `dcgm-custom-metrics.csv` 中定义的自定义指标，在端口 `9400` 暴露指标。

**前置条件：**
- 已安装 NVIDIA GPU 驱动
- Docker 已配置 NVIDIA Container Toolkit（`nvidia-docker2` 或 CDI）
- 需要存在 `xinference` Docker 网络：`docker network create xinference`

### 3. 配置 Prometheus

添加以下抓取目标：

- **Xinference Supervisor/Worker**（Xinference metrics 端点的默认端口）
- **DCGM Exporter**：`<worker-host>:9400`
- **node_exporter**：`<host>:9100`（用于主机资源面板）

## 面板详情

### 1. 集群概览（`xinference-overview`）

集群健康状态、Supervisor API 质量、告警。

| 面板 | 关键指标 |
|------|----------|
| Supervisor 运行时间 | `supervisor_uptime_seconds` |
| 在线 Worker 数 | `workers_total` |
| 按类型统计模型 | `models_loaded_total` |
| 模型异常终止 | `model_unexpected_termination` |
| API QPS / 错误 | `supervisor_http_requests_total` |
| 当前告警 | Alertmanager 集成 |

### 2. 模型负载（`xinference-model-load`）

以模型为中心的每模型/每副本视图，**覆盖所有模型类型**（LLM、embedding、rerank、image、audio、video）。将请求负载与显存占用统一展示。

| 面板 | 关键指标 |
|------|----------|
| 每类型 QPS / P95 / 错误率 | `model_request_total`、`model_request_duration_seconds` |
| 每类型总显存 | `model_gpu_memory_used_bytes` |
| 每模型总览表 | QPS、错误率、P95、并发、显存、GPU 绑定 |
| 每副本明细表 | 每 (model, worker) 的 QPS、并发、显存 |
| 趋势图 | QPS、并发率、显存、P95、错误率随时间变化 |

> **标签对齐：** Worker 侧和 Supervisor 侧指标现在都使用统一的标签（`model_uid`、`worker_address`、`model_type` 和 `replica_index`）。这允许在面板内直接进行 PromQL 关联（例如 `* on (model_uid, replica_index, worker_address)`），而不再需要 Grafana Transform 转换。此方式要求 Worker 和 Supervisor 指标位于**同一 Prometheus 实例**中（不同 `job` 标签）。

### 3. LLM 推理 SLO（`xinference-llm-slo`）

LLM 专属质量指标（仅 `model_type="LLM"`）。不含每模型明细（已迁移到模型负载面板）。

| 面板 | 关键指标 |
|------|----------|
| TTFT P50/P95/P99 | `time_to_first_token_seconds_bucket` |
| 请求延迟 P50/P95/P99 | `model_request_duration_seconds_bucket` |
| Token 吞吐 | `generate_tokens_total`、`input_tokens_total_counter`、`output_tokens_total_counter` |
| Token 放大比 | 输出 / 输入 Token |

### 4. GPU 资源（`xinference-gpu-resources`）

Worker 级 GPU 利用率与 DCGM 硬件健康。模型级面板已迁移到模型负载面板。

| 面板 | 关键指标 |
|------|----------|
| GPU 利用率 | `worker_gpu_utilization_percent` 或 `DCGM_FI_DEV_GPU_UTIL` |
| GPU 显存 | `worker_gpu_memory_used_bytes` |
| 各 Worker 活跃请求数 | `model_serve_count` |
| DCGM：温度、功耗、PCIe 错误、时钟、健康状态 | DCGM Exporter 指标 |

### 5. 主机资源（`xinference-host-resources`）

Supervisor 和 Worker 主机级指标。同时展示 Xinference 原生 gauge 和 node_exporter 指标。

| 面板 | 关键指标 |
|------|----------|
| CPU 利用率 | `worker_cpu_utilization`（Xinference）+ `node_cpu_seconds_total`（node_exporter） |
| 内存 | `worker_memory_used_bytes` / `worker_memory_total_bytes`（Xinference）+ `node_memory_*`（node_exporter） |
| 磁盘、Swap、网络、I/O | node_exporter 指标 |

### 6. 安全审计（`xinference-security-audit`）

API Key 与封禁统计。**仅在 `XINFERENCE_AUTH_ADVANCED=true` 时可用。**

| 面板 | 关键指标 |
|------|----------|
| 活跃 / 过期 API Key | `api_keys_active_total`、`api_keys_expired_total` |
| 封禁 IP / (IP, Key) 对 | `banned_ips_total`、`banned_keys_total` |
| API Key 请求 QPS / 延迟 | `api_key_requests_total`、`api_key_request_duration_seconds` |
| Top 用户 / Top 模型 | 按 API Key 请求量排序 |

## DCGM 自定义指标

`dcgm-custom-metrics.csv` 包含：

- **基础指标**：GPU 利用率、显存、温度、功耗、PCIe 错误、编码器/解码器使用率
- **增强指标**：功率上限、风扇转速、XID 错误、性能状态、时钟频率、PCIe 带宽、性能分析（Volta+ 架构）

## 环境要求

- Grafana >= 9.0
- Prometheus 数据源（Worker 和 Supervisor 指标须在同一实例中）
- Xinference 已启用 metrics
- DCGM Exporter，用于 GPU 硬件面板
- node_exporter，用于主机资源面板
