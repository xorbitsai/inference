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
- **表达式**: `rate(xinference:model_request_errors_total[5m]) / rate(xinference:model_request_total[5m]) > 0.05`

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

### RequestQueueBacklog — 请求队列积压

- **级别**: warning（警告）
- **条件**: 排队请求数 > 10，持续 3 分钟
- **表达式**: `xinference:model_pending_requests > 10`

### ModelLoadSlow — 模型加载缓慢

- **级别**: warning（警告）
- **条件**: 模型加载耗时 > 5 分钟
- **表达式**: `xinference:model_last_load_duration_seconds > 300`

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
