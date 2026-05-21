# Xinference Grafana 监控面板

预构建的 Grafana 监控面板，用于监控 Xinference 集群，涵盖 Supervisor API、Worker 资源、GPU 硬件（通过 DCGM）、模型服务质量及主机指标（通过 node_exporter）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `xinference-grafana-dashboard-en.json` | 英文版面板 |
| `xinference-grafana-dashboard-ja.json` | 日文版面板 |
| `xinference-grafana-dashboard-ko.json` | 韩文版面板 |
| `xinference-grafana-dashboard-zh-CN.json` | 精调中文版面板 |
| `dcgm-custom-metrics.csv` | 自定义 DCGM Exporter 指标配置 |
| `dcgm-exporter.yml` | DCGM Exporter 的 Docker Compose 服务定义 |

## 快速开始

### 1. 导入面板到 Grafana

1. 打开 Grafana → Dashboards → Import
2. 上传或粘贴所需语言的面板文件内容（如 `xinference-grafana-dashboard-zh-CN.json`）
3. 选择你的 Prometheus 数据源
4. 点击 Import

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

## DCGM 自定义指标

`dcgm-custom-metrics.csv` 包含：

- **基础指标**：GPU 利用率、显存、温度、功耗、PCIe 错误、编码器/解码器使用率
- **增强指标**：功率上限、风扇转速、XID 错误、性能状态、时钟频率、PCIe 带宽、性能分析（Volta+ 架构）

## 面板分区

| 分区 | 说明 |
|------|------|
| 集群概览 | Supervisor 运行时间、在线 Worker 数、构建/配置信息、按类型分类的模型数 |
| Supervisor API 监控 | API QPS、异常数、HTTP 错误率、状态码分布、Top 路径 |
| Worker 资源 | GPU 利用率、GPU 显存、CPU、内存 |
| GPU 硬件（DCGM） | 温度、功耗/功率上限、PCIe 错误、时钟频率、健康状态 |
| 模型状态与 GPU 绑定 | 模型部署表、GPU 绑定关系 |
| 模型服务质量 | QPS、错误率、P95 延迟、并发请求数 |
| LLM 专项 | 生成吞吐量（tokens/s）、首 Token 延迟 |
| Supervisor 主机资源 | CPU、内存、磁盘、网络、Swap、I/O（通过 node_exporter） |
| Worker 主机资源 | CPU、内存、磁盘、网络、Swap、I/O（通过 node_exporter） |

## 环境要求

- Grafana >= 9.0
- Prometheus 数据源
- Xinference 已启用 metrics
- DCGM Exporter，用于 GPU 硬件面板
- node_exporter，用于主机资源面板
