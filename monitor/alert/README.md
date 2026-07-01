# Xinference Alert Rules

Prometheus alert rules for monitoring Xinference cluster health.

## Files

| File | Language |
|------|----------|
| `rules.yml` | English |
| `rules-zh-CN.yml` | 中文 |
| `rules-ja.yml` | 日本語 |
| `rules-ko.yml` | 한국어 |

All files share the same alert expressions and thresholds — only `summary` and `description` annotations are localized.

## Alert Rules

### GPUMemoryHigh

- **Severity**: critical
- **Condition**: GPU memory usage > 90% for 5 minutes
- **Expression**: `xinference:worker_gpu_memory_used_bytes / xinference:worker_gpu_memory_total_bytes > 0.9`

### ModelRequestErrorRateHigh

- **Severity**: warning
- **Condition**: Model request error rate > 5% for 3 minutes
- **Expression**: `sum without (stream) (rate(xinference:model_request_errors_total[5m])) / sum without (stream) (rate(xinference:model_request_total[5m])) > 0.05`

### TTFTHigh

- **Severity**: warning
- **Condition**: Time to first token > 10 seconds for 3 minutes
- **Expression**: `rate(xinference:time_to_first_token_seconds_sum[5m]) / rate(xinference:time_to_first_token_seconds_count[5m]) > 10`

### WorkerOffline

- **Severity**: critical
- **Condition**: Online worker count < 1 for 2 minutes
- **Expression**: `xinference:workers_total < 1`

### DiskSpaceLow

- **Severity**: critical
- **Condition**: Disk available space < 10% for 5 minutes
- **Expression**: `(node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1`
- **Note**: This rule depends on [node_exporter](https://github.com/prometheus/node_exporter) being deployed on all cluster nodes.

### RequestQueueBacklog

- **Severity**: warning
- **Condition**: Model concurrency ratio (active / limit) > 90% for 3 minutes
- **Expression**: `xinference:model_serve_count / (xinference:model_request_limit > 0) > 0.9`

### ModelLoadSlow

- **Severity**: warning
- **Condition**: Model load time > 5 minutes
- **Expression**: `xinference:model_last_load_duration_seconds > 300`

### ReplicaUnexpectedTerminated

- **Severity**: critical
- **Condition**: A replica is marked as unexpectedly terminated (worker failure) for 1 minute
- **Expression**: `xinference:model_unexpected_termination == 1`

### WorkerMemoryHigh

- **Severity**: warning
- **Condition**: Worker memory usage > 90% for 5 minutes
- **Expression**: `xinference:worker_memory_used_bytes / xinference:worker_memory_total_bytes > 0.9`

### RequestLatencyHigh

- **Severity**: warning
- **Condition**: P95 request latency > 60 seconds for 3 minutes
- **Expression**: `histogram_quantile(0.95, rate(xinference:model_request_duration_seconds_bucket[5m])) > 60`

### BannedIPsSpike

- **Severity**: warning
- **Condition**: Banned IP count > 10 for 2 minutes
- **Expression**: `xinference:banned_ips_total > 10`

## Usage

Add the rule file to your Prometheus configuration:

```yaml
# prometheus.yml
rule_files:
  - "/path/to/monitor/alert/rules.yml"
```

Reload Prometheus to apply:

```bash
curl -X POST http://localhost:9090/-/reload
```

## Alertmanager Integration

Configure Alertmanager to route alerts by severity:

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
