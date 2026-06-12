# Xinference 告警规则

用于监控 Xinference 集群健康状态的 Prometheus 告警规则。

## 文件说明

| 文件 | 语言 |
|------|------|
| `rules.yml` | English |
| `rules-zh-CN.yml` | 中文 |
| `rules-ja.yml` | 日本語 |
| `rules-ko.yml` | 한국어 |

所有文件使用相同的告警表达式和阈值，仅 `summary` 和 `description` 注解做了本地化翻译。

## 告警规则

### GPUMemoryHigh — GPU 显存使用率过高

- **级别**: critical（严重）
- **条件**: GPU 显存使用率 > 90%，持续 5 分钟
- **表达式**: `xinference:worker_gpu_memory_used_bytes / xinference:worker_gpu_memory_total_bytes > 0.9`

### ModelRequestErrorRateHigh — 模型请求错误率过高

- **级别**: warning（警告）
- **条件**: 模型请求错误率 > 5%，持续 3 分钟
- **表达式**: `sum without (stream) (rate(xinference:model_request_errors_total[5m])) / sum without (stream) (rate(xinference:model_request_total[5m])) > 0.05`

### TTFTHigh — 首 Token 延迟过高

- **级别**: warning（警告）
- **条件**: 首 Token 延迟 > 10 秒，持续 3 分钟
- **表达式**: `rate(xinference:time_to_first_token_seconds_sum[5m]) / rate(xinference:time_to_first_token_seconds_count[5m]) > 10`

### WorkerOffline — Worker 节点离线

- **级别**: critical（严重）
- **条件**: 在线 Worker 数量 < 1，持续 2 分钟
- **表达式**: `xinference:workers_total < 1`

### DiskSpaceLow — 磁盘空间不足

- **级别**: critical（严重）
- **条件**: 磁盘可用空间 < 10%，持续 5 分钟
- **表达式**: `(node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1`
- **备注**: 此规则依赖在所有集群节点上部署 [node_exporter](https://github.com/prometheus/node_exporter)。

### RequestQueueBacklog — 请求并发接近上限

- **级别**: warning（警告）
- **条件**: 模型并发比率（活跃数 / 上限）> 90%，持续 3 分钟
- **表达式**: `xinference:model_serve_count / (xinference:model_request_limit > 0) > 0.9`

### ModelLoadSlow — 模型加载缓慢

- **级别**: warning（警告）
- **条件**: 模型加载耗时 > 5 分钟
- **表达式**: `xinference:model_last_load_duration_seconds > 300`

### ReplicaUnexpectedTerminated — 副本异常终止

- **级别**: critical（严重）
- **条件**: 副本因 Worker 故障被标记为异常下线，持续 1 分钟
- **表达式**: `xinference:model_unexpected_termination == 1`

### WorkerMemoryHigh — Worker 内存使用率过高

- **级别**: warning（警告）
- **条件**: Worker 内存使用率 > 90%，持续 5 分钟
- **表达式**: `xinference:worker_memory_used_bytes / xinference:worker_memory_total_bytes > 0.9`

### RequestLatencyHigh — 请求延迟过高

- **级别**: warning（警告）
- **条件**: P95 请求延迟 > 60 秒，持续 3 分钟
- **表达式**: `histogram_quantile(0.95, rate(xinference:model_request_duration_seconds_bucket[5m])) > 60`

### BannedIPsSpike — 封禁 IP 数量异常

- **级别**: warning（警告）
- **条件**: 封禁 IP 数 > 10，持续 2 分钟
- **表达式**: `xinference:banned_ips_total > 10`

## 使用方法

在 Prometheus 配置中添加规则文件：

```yaml
# prometheus.yml
rule_files:
  - "/path/to/monitor/alert/rules-zh-CN.yml"
```

重新加载 Prometheus 使配置生效：

```bash
curl -X POST http://localhost:9090/-/reload
```

## Alertmanager 集成

配置 Alertmanager 按告警级别进行路由分发：

```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  receiver: default
  routes:
    - match:
        severity: critical
      receiver: critical-channel
    - match:
        severity: warning
      receiver: warning-channel
```
